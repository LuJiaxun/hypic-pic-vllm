# PIC 异步导入（W1 / C1c）调用流程

按代码实际路径记录，用于定位"请求为什么没被采纳"这类时序问题。行号只作参考，函数名是稳定锚点。

## 一次 park 的完整生命周期

```
① scheduler.add_request(request)
     request.pic_cache_plan = pic_cache.get_computed_blocks(...)   # lookup，matches 里带 external refs
     _ensure_pic_public_blocks(request)                            # external 段分配共享 public blocks
                                                                    # → request.pic_public_block_ids

② scheduler.schedule() 等待队列循环
     new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
     _park_pic_external_import(request, block_ids)
         ├─ 已导入过（_pic_import_completed）        → 不 park
         ├─ 没有 external refs                       → 不 park
         ├─ 最后一个可运行请求（防死锁）              → 不 park（走 inline）
         └─ 否则：_pic_pending_imports[req_id] = PICAsyncImportSpec(request_id,
                                                                   segment_indices,
                                                                   NewRequestData(...))
                  status = WAITING_FOR_REMOTE_KVS
                  step_skipped_waiting.prepend_request(request); continue

③ scheduler 组装 SchedulerOutput
     scheduler_output.pic_async_imports = tuple(_pic_pending_imports.values())

④ worker.execute_model() 开头 → _pic_async_import_step(scheduler_output)
     先 poll 上一批：_pic_park_poll()
         batch.poll() / transfer.poll_state() → ready → manager.complete(req_id, payload)
                                              → failed → manager.fail(req_id, reason)
     再 submit 新的：_pic_park_submit(spec)
         _prepare_pic_mooncake_destinations(stand_in)   # 用 public blocks 排队目的地
         begin_import_references(refs)                  # 后台线程开始传输
         manager.start(spec, payload)

⑤ worker.execute_model() 内 → _update_states(scheduler_output)
     对 scheduled_new_reqs 里的每个请求（被 park 的请求此时首次进入）：
         req_state = CachedRequestState(..., pic_enabled, pic_cache_plan, ...)
         _set_pic_runtime_plan(req_state)
             _release_pic_request_leases(req_state)
             _release_pic_imported_native_refs(req_state)   # 先清旧引用
             _pic_adopt_park_import(req_state)              # ★ 采纳点（必须在清理之后）
             _build_pic_runtime_plan(...)                   # 用刚装好的 refs，跳过 import

⑥ worker 输出 ModelRunnerOutput
     pic_finished_imports / pic_failed_imports
     空步（num_scheduled_tokens == 0）也要带：_pic_with_import_reports(EMPTY_MODEL_RUNNER_OUTPUT)

⑦ scheduler.update_from_output → _update_pic_async_imports(model_output)
     finished → _pic_finished_recving.add(req_id)
     failed   → _pic_failed_recving.add(req_id)

⑧ 下一步 scheduler.schedule() 循环顶部
     _try_promote_blocked_waiting_request(request)
         READY  ：_pic_import_completed.add(req_id); status = WAITING; 清 pending
         FAILED ：释放 blocks、plan=None、status = WAITING（全量重算）

⑨ 同一循环继续 → allocate_slots → _park_pic_external_import 因 _pic_import_completed 返回 False
     → scheduled_new_reqs.append(request) → 回到 ⑤（adopt 命中）
```

## 采纳点为什么只有一个位置

| 位置 | 是否会执行 | 作用 |
|---|---|---|
| `_set_pic_runtime_plan` 内，`_release_pic_imported_native_refs` 之后 | **每条请求必经** | 主路径（唯一采纳点） |
| `_build_pic_runtime_plan` 开头 | 可能被早退条件跳过 | 幂等兜底 |

两个坑都踩过：

1. 只在 `_build_pic_runtime_plan` 里采纳 —— 该函数有 5 个早退条件（`pic_enabled`、
   `pic_worker_plan`、capacity、`native_kv_bridge`、`pic_cache_plan`），任何一个不满足都会让
   采纳静默不发生（`ADOPT=0`）。
2. 在 `_set_pic_runtime_plan` **之前**采纳 —— 该函数开头的
   `_release_pic_imported_native_refs` 会把刚装好的引用全部释放并清空，于是计划构建时又走一遍
   inline import，同一份 KV 被导入两次（收益被吃掉一半）。

因此采纳点的正确位置是"清理旧引用之后、构建计划之前"。

## 状态集合一览

| 集合 | 写入 | 读取 | 含义 |
|---|---|---|---|
| `_pic_pending_imports` | park | 组装 SchedulerOutput、promote 时清 | 已发出但未完成的 spec |
| `_pic_finished_recving` | `_update_pic_async_imports` | promote | worker 报告导入完成 |
| `_pic_failed_recving` | `_update_pic_async_imports` | promote | worker 报告导入失败 |
| `_pic_import_completed` | promote READY | park 判定、free 时清理 | 已导入，不再 park |
| worker `_pic_async_imports`（manager） | submit/complete | poll、adopt | 请求级导入状态与结果 |
| worker `_pic_import_reports` | poll | 输出构造 | 待上报的 finished/failed |
| `_pic_public_shared_holders` | 授予 | 请求释放、淘汰 | 共享块引用计数（缓存 1 + 每请求 1） |

## 已知缺口

| 缺口 | 影响 |
|---|---|
| 共享集字节未计入 `max_cache_bytes` | 钉住的 canonical blocks 不参与 LRU 预算，长跑可能超预算 |
| lease finalize 后无 tombstone | 同 key 迟到的 import 可能拿到"新租约但底层已释放" |

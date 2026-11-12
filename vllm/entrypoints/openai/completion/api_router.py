# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.v1.engine.pic_profile import emit as pic_profile_emit
from vllm.v1.engine.pic_profile import start as pic_profile_start

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    request_start_ns = pic_profile_start()
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = completion(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions API")

    request_id = f"cmpl-{handler._base_request_id(raw_request, request.request_id)}"
    raw_request.state.pic_profile_request_id = request_id
    pic_profile_emit(
        request_id,
        "unknown",
        "api.request_received",
        request_start_ns,
        request_start_ns,
    )
    handler_start_ns = pic_profile_start()
    handler_create_start_ns = pic_profile_start()
    generator = await handler.create_completion(request, raw_request)
    pic_profile_emit(
        request_id,
        "unknown",
        "api.handler_create_completion",
        handler_create_start_ns,
    )
    pic_profile_emit(
        request_id,
        "unknown",
        "api.handler_total",
        handler_start_ns,
    )

    if isinstance(generator, ErrorResponse):
        serialize_start_ns = pic_profile_start()
        response = JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
        pic_profile_emit(
            request_id, "unknown", "api.response_serialize", serialize_start_ns
        )
        return response
    elif isinstance(generator, CompletionResponse):
        serialize_start_ns = pic_profile_start()
        response = JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )
        pic_profile_emit(
            request_id, "unknown", "api.response_serialize", serialize_start_ns
        )
        return response

    return StreamingResponse(content=generator, media_type="text/event-stream")


def attach_router(app: FastAPI):
    app.include_router(router)

import os
import time

from fastapi import APIRouter, File, Form, Response

from .. import function_wrapper
from .. import config

api_router = APIRouter()
runner = function_wrapper.FunctionWrapper(config)

@api_router.get('/welcome/')
async def welcome() -> dict:
    result_dict = dict()
    result_dict['message'] = 'WELCOME TO PDFs Chatbot'
    result_dict['current_collection'] = runner.collection_name
    return result_dict



@api_router.post('/user_ask/')
async def user_question(
        question: str=Form(...),
        session_name: str=Form(...),
    ):
    result_dict = dict()
    t_1 = time.time()
    runner.reload_retrieval(collection_name=session_name)
    t_2 = time.time()
    result_dict['answer'] = runner.answer_query(question)
    t_3 = time.time()
    result_dict['time_reload_retrieval'] = str(round(t_2-t_1, 4))
    result_dict['time_answer'] = str(round(t_3-t_2, 4))
    return result_dict


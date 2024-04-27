import shutil
import os
from os.path import join as opj
import time
import uuid

from fastapi import APIRouter, File, UploadFile, Form, Response
from loguru import logger

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


# @api_router.post('/create_collection')
# async def create_collection(
#     collection_name=None,
#     user_name: str=Form(...)
#     ) -> dict:
#     if collection_name is None:
#         collection_name = str(uuid.uuid4())
#     storage_name = opj(user_name, collection_name)
#     storage_path = opj(os.getcwd(), 'chatbot_util', 'media', storage_name)
#     os.makedirs(storage_path, exist_ok=True)
#     result = os.path.isdir(storage_path)
#     return {'message': str(result)}



@api_router.post('/collection/delete_collection')
async def delete_collection(
    collection_name: str=Form(...),
    user_name: str=Form(...)
    ) -> dict:
    file_path = opj(os.getcwd(), 'chatbot_util', 'media', user_name, collection_name)
    if os.path.isdir(file_path):
        # remove physical file
        logger.info('remove {}'.format(file_path))
        shutil.rmtree(file_path)
        # remove collection in vectorstore
        runner.vectorDB_client.delete_collection(collection_name)
        return {'message': 'success'}
    else:
        return {'message': 'collection does not exist'}


@api_router.post("/collection/uploadfiles/")
async def create_upload_files(
    files: list[UploadFile],
    user_name: str=Form(...),
    collection_name: str=Form(...)
    ):
    # create collection folder whether it exist or not
    store_path = opj(os.getcwd(), 'chatbot_util', 'media', user_name, collection_name)
    os.makedirs(store_path, exist_ok=True)
    file_paths = []
    for file in files:
        file_path = opj(store_path, file.filename)
        file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
    for path in file_paths:
        runner.emb_document(path, user_name)
    runner.reload_retrieval(collection_name=collection_name)
    return {"filenames": [file.filename for file in files]}


@api_router.post('/collection/deletefiles/')
async def delete_files(
    files_name: str=Form(...),
    user_name: str=Form(...),
    collection_name: str=Form(...)
    ):
    return {
        'file_names': files_name,
        'user_names': user_name,
        'collection_name': collection_name
    }

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


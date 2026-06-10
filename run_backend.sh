#!/bin/bash
# 启动后端（LangGraph + FastAPI）
cd backend
pip install -e .
langgraph dev

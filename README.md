# Build a Multi-Agent System with LangGraph 🦜🕸️

## Overview

This notebook demonstrates how to build and train a multi-agent system using LangGraph and LangChain. The system is designed to handle customer service queries for a digital music store through coordinated interactions between specialized agents.

## System Architecture

The system consists of three specialized subagents:

1. **Customer Information Agent**: Handles customer profile data (name, email, address, etc.)
2. **Music Catalog Agent**: Manages queries about the music store's catalog (albums, tracks, artists)
3. **Invoice Information Agent**: Processes customer purchase history and invoice queries

## Key Features

- **Supervisor/Planner Architecture**: Uses a main agent to coordinate subagents
- **Memory Management**: Maintains customer context across conversations
- **Human-in-the-Loop**: Supports user intervention and plan modification
- **Efficient Task Distribution**: Optimizes task allocation across subagents

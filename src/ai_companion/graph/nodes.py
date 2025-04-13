import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from ai_companion.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.settings import settings


async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})  ### giving llm last few (5) messages to analyze
    return {"workflow": response.response_type}  ## response_type is either conversation, image or audio


def context_injection_node(state: AICompanionState):
    schedule_context = ScheduleContextGenerator.get_current_activity()   ## given the current time, it gets the activity Ava is doing at that time
    if schedule_context != state.get("current_activity", ""):  ## if the current activity is different from the last one, then apply_activity is True
        apply_activity = True
    else:
        apply_activity = False
    return {"apply_activity": apply_activity, "current_activity": schedule_context}


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")   ## get the memory context from the state  ( return memory_context from wrokflow graph and "" if not present)

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response)}


async def image_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")   ## get the memory context from the state  ( return memory_context from wrokflow graph and "" if not present)

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(content=f"<image attached by Ava generated from prompt: {scenario.image_prompt}>")
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}


async def audio_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")   ## get the memory context from the state  ( return memory_context from wrokflow graph and "" if not present)

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    output_audio = await text_to_speech_module.synthesize(response)

    return {"messages": response, "audio_buffer": output_audio}


async def summarize_conversation_node(state: AICompanionState):
    model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Ava and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Ava and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Ava and the user:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages}


async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message."""
    if not state["messages"]:    #### check if there are any messages in the state
        return {}

    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1])  ## If messages exist, it gets the most recent message with state["messages"][-1]  and passes to memory_manager.extract_and_store_memories()
    return {}


def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the character card."""
    memory_manager = get_memory_manager()

    # Get relevant memories based on recent conversation
    recent_context = " ".join([m.content for m in state["messages"][-3:]])  ## get the last 3 messages to create a context for the memories
    memories = memory_manager.get_relevant_memories(recent_context)   ## get_relevant_memories() returns a list of relevant memories based on the recent context

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}




# Overall Architecture
# Your application is built using LangGraph to create a directed workflow graph where user messages flow through different processing nodes. The main components are:

# State Management: AICompanionState tracks conversation history and other data
# Workflow Graph: Defines how messages flow through the system
# Processing Nodes: Handle specific tasks like memory management, response generation, etc.
# External Services: Groq for LLMs, ElevenLabs for speech, etc.

# Complete Workflow
# When a user interacts with the AI companion, here's what happens step by step:
# 1. User Sends a Message

# The user message is added to the state's message history

# 2. Memory Extraction (First Node)

# memory_extraction_node analyzes the message for important personal information
# The message is sent to memory_manager.extract_and_store_memories()
# Important information is identified, formatted, and stored in a vector database

# 3. Route Selection

# router_node examines recent messages to determine what type of response is needed
# It uses get_router_chain() to decide between "conversation", "image", or "audio"
# The decision is stored in the state's workflow attribute

# 4. Context Injection

# context_injection_node gets Ava's current activity based on the time of day
# It pulls the appropriate activity from the schedule (e.g., "working at Groq", "at an art gallery")
# This activity is stored in the state's current_activity attribute

# 5. Memory Injection

# memory_injection_node retrieves relevant memories about the user
# It looks at recent messages and finds matching memories in the vector database
# These memories are formatted and added to the state's memory_context

# 6. Response Generation (Three Possible Paths)
# Based on the earlier routing decision:

# Conversation Node: Generates a text response using the character prompt
# Image Node: Creates a visual scenario and generates an image, then responds
# Audio Node: Generates a text response and converts it to speech

# Each node incorporates:

# The message history
# Ava's current activity
# Relevant memories about the user

# 7. Conversation Summarization (Optional)

# After a certain number of messages (TOTAL_MESSAGES_SUMMARY_TRIGGER)
# summarize_conversation_node creates a summary of the conversation
# Old messages are removed, summary is stored for context preservation

# 8. Response Delivered to User

# The final response (text, image, or audio) is sent back to the user

# Key Files and Their Roles

# graph.py: Creates the workflow graph and defines connections between nodes
# nodes.py: Contains all the processing functions for each step
# state.py: Defines what information is tracked throughout the conversation
# chains.py: Creates the language model chains for different tasks
# prompts.py: Contains the system prompts for different operations
# memory_manager.py: Handles storing and retrieving user information
# vector_store.py: Interfaces with Qdrant for persistent memory storage
# schedules.py: Contains Ava's daily activities for context
# settings.py: Global configuration for models and parameters









##### How to store data in Workflow Graphs
## METHOD 1
# async def my_node(state: AICompanionState):
#     # Do some processing
#     result = "Some processed data"
    
#     # Return only the keys you want to update
#     return {"my_custom_data": result}

### METHOD 2
# async def my_node(state: AICompanionState):
#     # Direct modification (not typically recommended)
#     state["my_custom_data"] = "Some value"
    
#     # Return empty dict or the modified keys
#     return {}  # or return {"my_custom_data": state["my_custom_data"]}





#### HOW TO GET THE VALUES FROM THE WORKFLOW GRAPH
# def my_edge_function(state: AICompanionState):
#     if state["workflow"] == "image":
#         return "image_node"
#     elif state["workflow"] == "audio":
#         return "audio_node"
#     else:
#         return "conversation_node"






















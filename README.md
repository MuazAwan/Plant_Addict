# Re:amaze Automated Customer Follow-Up System

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated system designed to enhance customer engagement for e-commerce businesses using the Re:amaze platform. This script intelligently sends timed, AI-generated follow-up messages to potential customers who have shown buying intent but haven't yet completed a purchase.

## Overview

In the fast-paced e-commerce environment, timely and relevant follow-ups can significantly impact conversion rates. This project automates the process of re-engaging potential customers who have interacted with support (e.g., via Re:amaze chat) but haven't converted. It aims to provide a personalized touch without manual agent intervention for initial follow-ups, freeing up support teams to handle more complex inquiries.

The system polls Re:amaze for conversations manually tagged by agents (indicating sales potential), then orchestrates a two-step follow-up sequence with configurable delays, purchase checks, and AI-generated message content.

## Features

*   **Automated Polling:** Regularly checks Re:amaze via API for newly tagged conversations.
*   **Configurable Tag Trigger:** Initiates the workflow based on a specific tag applied by agents.
*   **Timed Follow-Ups:** Sends two distinct follow-up messages at configurable intervals (e.g., 30 minutes and 24 hours).
*   **Purchase Detection:** Checks conversation history for purchase indicators before sending follow-ups to avoid messaging converted customers.
*   **AI-Generated Messages:** Leverages Google's Gemini API (e.g., `gemini-1.5-flash-latest`) to generate context-aware, natural-sounding follow-up messages based on the prior conversation history.
*   **Fallback System:** Uses pre-defined fallback messages if AI generation fails.
*   **State Management:** Utilizes an SQLite database to reliably track the state of each conversation in the follow-up workflow, preventing duplicate processing.
*   **Robust Error Handling & Retries:** Includes API request retries for transient network issues and comprehensive logging.
*   **Externalized Configuration:** API keys, prompts, operational parameters, and message templates are managed via `.env` and `config.json` files for easy customization without code changes.
*   **Scheduled Operation:** Uses APScheduler for managing internal job scheduling.
*   **(Future Scope - Currently Manual):** Logic for automatic removal of the trigger tag from Re:amaze upon workflow completion (pending API clarification/implementation).

## Tech Stack

*   **Language:** Python 3.9+
*   **Key Libraries:**
    *   `requests`: For making HTTP API calls to Re:amaze.
    *   `APScheduler`: For scheduling polling and processing jobs.
    *   `python-dotenv`: For managing environment variables (API keys, secrets).
    *   `google-generativeai`: For interacting with Google Gemini LLMs.
    *   `sqlite3`: For persistent state management.
*   **APIs:**
    *   Re:amaze REST API (v1)
    *   Google Generative AI API (Gemini)
*   **Configuration:** JSON (`config.json`), Environment Variables (`.env`)

## Architecture Overview

1.  **Scheduler (`APScheduler`):**
    *   A job (`check_for_new_conversations`) runs at a configured interval (e.g., every 15 minutes) to poll the Re:amaze API.
    *   Another job (`process_conversations`) runs more frequently (e.g., every 5 minutes) to process the state machine.
2.  **Polling & State Initiation (`check_for_new_conversations`):**
    *   Fetches conversations from Re:amaze tagged with a specific trigger tag.
    *   Compares found conversations against an SQLite database.
    *   Adds newly tagged, untracked conversations to the database with an initial status (e.g., `SCHEDULED_CHECK_1`) and an `added_time`.
3.  **Workflow Processing (`process_conversations`):**
    *   Retrieves conversations from the database that are in an active (non-final) state.
    *   For each conversation:
        *   **Wait 1 (e.g., 30 mins):** If status is `SCHEDULED_CHECK_1` and `check1_time` (calculated from `added_time` + wait duration) is due:
            *   Fetches latest messages from Re:amaze.
            *   Checks for purchase indicators.
            *   If purchase: Updates status to `COMPLETED_PURCHASE_DETECTED_1`.
            *   If no purchase: Generates Follow-up #1 message using Gemini (based on conversation history and prompt from `config.json`). Sends message to Re:amaze. Updates status to `SCHEDULED_CHECK_2` and sets `check2_time`.
        *   **Wait 2 (e.g., 24 hrs):** If status is `SCHEDULED_CHECK_2` and `check2_time` is due:
            *   Fetches latest messages.
            *   Checks for purchase.
            *   If purchase: Updates status to `COMPLETED_PURCHASE_DETECTED_2`.
            *   If no purchase: Generates Follow-up #2 message using Gemini (based on history and prompt). Sends message. Updates status to `COMPLETED_NO_PURCHASE`.
        *   All state changes and timestamps are recorded in the SQLite database.
4.  **LLM Interaction:** Specific prompts guide the Gemini model to generate contextually relevant messages. Fallback messages are used if AI generation fails.

## Setup & Installation

### Prerequisites

*   Python 3.9 or higher.
*   `pip` (Python package installer).
*   Access to a Re:amaze account with API capabilities.
*   A Google Cloud Project with the "Generative Language API" (or Vertex AI) enabled and an API key.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MuazAwan/Plant_Addict.git
    cd Plant_Addict
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables (`.env` file):**
    Create a `.env` file in the project root directory with the following content, replacing placeholder values:
    ```dotenv
    REAMAE_SUBDOMAIN=your_reamaze_subdomain # e.g., plant-addicts
    REAMAE_LOGIN_EMAIL=your_reamaze_login_email_for_api
    REAMAE_API_TOKEN=your_reamaze_api_token
    TRIGGER_TAG_NAME=Needs-Sales-Followup # Or your chosen trigger tag
    GOOGLE_API_KEY=your_google_cloud_api_key_for_gemini

    # Optional: For development/testing (comment out for production defaults)
    # WAIT_1_MINUTES=3 
    # WAIT_2_MINUTES=5
    # LOG_LEVEL=DEBUG 
    ```

5.  **Configure Operational Settings (`config.json`):**
    Create a `config.json` file in the project root directory. Refer to the example below and customize prompts, messages, and intervals as needed:
    ```json
    {
      "followup1_fallback_message": "Hi there! Just circling back on our recent chat. Please let me know if you have any more questions!",
      "followup2_fallback_message": "Hey again! Checking in on our previous conversation. Please let me know if I can help with anything else!",
      "llm_prompt_followup1": "You are 'Rizwan' from Plant Addicts... Conversation:\n{conversation_text}\n\nWrite only the follow-up message content:",
      "llm_prompt_followup2": "You are 'Rizwan' from Plant Addicts... Conversation:\n{conversation_text}\n\nWrite only the follow-up message content:",
      "followup1_helpful_links_header": "\n\nHelpful Resources:",
      "followup1_helpful_links": [
        {"text": "Plant Care Guides", "url": "https://plantaddicts.com/plant-care/"},
        {"text": "Shipping Information", "url": "https://plantaddicts.com/REPLACE_WITH_PUBLIC_SHIPPING_URL"},
        {"text": "Houseplants Collection", "url": "https://plantaddicts.com/houseplants"}
      ],
      "gemini_model": "gemini-1.5-flash-latest",
      "gemini_summary_max_output_tokens": 250,
      "gemini_summary_temperature": 0.5,
      "poll_interval_minutes": 15,
      "process_interval_minutes": 5,
      "max_api_retries": 3,
      "api_retry_delay_seconds": 5
    }
    ```
    *(Ensure prompts are fully defined as per your working version)*

6.  **Initialize the Database:**
    The script will automatically create the `workflow_state.db` SQLite file and the necessary table on its first run due to the `init_db()` call.

## Usage

1.  Ensure your `.env` and `config.json` files are correctly configured.
2.  Activate your virtual environment: `source .venv/bin/activate`
3.  Run the script:
    ```bash
    python app.py 
    ```
4.  The script will start logging to the console and `app.log`. It will begin polling Re:amaze for conversations tagged with your `TRIGGER_TAG_NAME`.
5.  To test, manually apply the trigger tag to a conversation in Re:amaze.
6.  The script will process the conversation according to the configured wait times and logic.
7.  Press `Ctrl+C` to stop the script.

### Deployment for Continuous Operation

For production, the script needs to run continuously on a server. Recommended platforms include:
*   **PythonAnywhere:** Use an "Always-on task" (Hacker plan or higher recommended for unrestricted API access and sufficient CPU).
*   **VPS (DigitalOcean, Linode, AWS EC2, GCP Compute Engine):** Requires manual setup of the environment and a process manager like `systemd` or `supervisor` to ensure the script runs persistently and restarts on failure/reboot.
*   **Other PaaS solutions** that support long-running Python background workers.

## Configuration Details

*   **`.env` File:**
    *   `REAMAE_SUBDOMAIN`: Your Re:amaze account subdomain.
    *   `REAMAE_LOGIN_EMAIL`: Email associated with the Re:amaze API token.
    *   `REAMAE_API_TOKEN`: Your Re:amaze API token.
    *   `TRIGGER_TAG_NAME`: The specific tag that triggers the automation.
    *   `GOOGLE_API_KEY`: Your Google Cloud API key for Gemini.
    *   `WAIT_1_MINUTES` (Optional): Overrides the default 30-minute wait for the first follow-up.
    *   `WAIT_2_MINUTES` (Optional): Overrides the default 24-hour wait for the second follow-up.
    *   `LOG_LEVEL` (Optional): Set to `DEBUG`, `INFO`, `WARNING`, `ERROR`. Defaults to `INFO`.
*   **`config.json` File:**
    *   `llm_prompt_followup1`/`llm_prompt_followup2`: Prompts for the Gemini model for each follow-up.
    *   `followup1_fallback_message`/`followup2_fallback_message`: Messages to send if AI generation fails.
    *   `followup1_helpful_links*`: Configuration for static helpful links (note: current Python code does not automatically append these after LLM response unless that logic is re-added).
    *   `gemini_model`: The specific Gemini model to use (e.g., `gemini-1.5-flash-latest`).
    *   `gemini_summary_max_output_tokens`: Max tokens for the LLM response.
    *   `gemini_summary_temperature`: Creativity setting for the LLM (0.0 to 1.0).
    *   `poll_interval_minutes`: How often to check Re:amaze for new tags.
    *   `process_interval_minutes`: How often to process the conversation states.
    *   `max_api_retries`: Number of retries for failed API calls.
    *   `api_retry_delay_seconds`: Seconds to wait between API retries.

## Future Enhancements

*   **Automatic Tag Removal:** Fully implement and test tag removal upon workflow completion once Re:amaze API details are confirmed or if a robust workaround is found.
*   **Dynamic Link Inclusion:** Enhance LLM prompts or add Python logic to intelligently select and include only the *most relevant* links from the conversation or a curated list.
*   **Advanced RAG for LLM:** Integrate a vector database with website/FAQ content to provide richer, factual context to the LLM for more informed responses.
*   **Admin Interface/Dashboard:** A simple web UI to monitor script status, view processed conversations, and manage configurations.
*   **More Granular Error Handling:** Implement more specific error states and potential automated recovery actions.
*   **Unit & Integration Testing:** Develop a comprehensive test suite.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file (you'll need to create this) for details.

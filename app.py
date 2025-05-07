# --- Start of app.py ---
import logging
import requests
import os
import base64
import time
import json
import traceback
import sqlite3 # Use SQLite
from datetime import datetime, timedelta, timezone

# --- Setup Logging ---
log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
log_format = '%(asctime)s - %(levelname)-8s - %(name)s - [%(module)s:%(lineno)d] - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)
try:
    log_file = 'app.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Logging configured. Level: {log_level_str}. Outputting to console and {log_file}")
except Exception as e:
    logging.error(f"Failed to configure file logging to {log_file}: {e}")
    logging.info(f"Logging configured. Level: {log_level_str}. Outputting to console only.")

logger = logging.getLogger(__name__)
logger.info("Script starting...")

# --- Load Configuration File ---
CONFIG_FILE = 'config.json'
APP_CONFIG = {}

def load_config():
    global APP_CONFIG
    logger.info(f"Loading configuration from {CONFIG_FILE}...")
    try:
        with open(CONFIG_FILE, 'r') as f:
            APP_CONFIG = json.load(f)
        logger.info("Operational config loaded successfully.")
        # Validate essential config keys for this version
        required_config = ['llm_prompt_followup1', 'llm_prompt_followup2',
                           'followup1_fallback_message', 'followup2_fallback_message']
        missing_config = [k for k in required_config if k not in APP_CONFIG]
        if missing_config:
            logger.critical(f"Missing required keys in {CONFIG_FILE}: {', '.join(missing_config)}")
            # Decide if exit() is needed here or handle with defaults later
            # For safety, let's exit if core prompts/fallbacks are missing
            exit()
        # Set defaults for optional operational configs if missing
        APP_CONFIG.setdefault('poll_interval_minutes', 15)
        APP_CONFIG.setdefault('process_interval_minutes', 5)
        APP_CONFIG.setdefault('max_api_retries', 3)
        APP_CONFIG.setdefault('api_retry_delay_seconds', 5)
        APP_CONFIG.setdefault('gemini_model', 'gemini-1.5-flash-latest')
        APP_CONFIG.setdefault('gemini_summary_max_output_tokens', 200)
        APP_CONFIG.setdefault('gemini_summary_temperature', 0.7)

    except FileNotFoundError:
        logger.critical(f"{CONFIG_FILE} not found. Cannot proceed without required configurations.")
        exit()
    except json.JSONDecodeError:
        logger.critical(f"Invalid JSON in {CONFIG_FILE}. Cannot proceed.")
        exit()
    except Exception as e:
         logger.exception(f"Unexpected error loading {CONFIG_FILE}. Cannot proceed.")
         exit()

# --- Check Essential Imports ---
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    logger.debug("APScheduler imported successfully.")
except ImportError as e:
    logger.critical(f"Failed to import APScheduler. Is 'apscheduler' installed? Error: {e}")
    exit()

try:
    from dotenv import load_dotenv
    logger.debug("Dotenv imported.")
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    logger.debug(f"Looking for .env file at: {dotenv_path}")
    loaded = load_dotenv(dotenv_path=dotenv_path)
    if loaded: logger.info(".env file loaded successfully.")
    else: logger.warning(".env file not found. Relying on system environment variables.")
except ImportError: logger.warning("python-dotenv not found.")
except Exception as e: logger.error(f"Error loading .env file: {e}")

try:
    import google.generativeai as genai
    logger.debug("Google Generative AI library imported successfully.")
except ImportError:
    logger.critical("Google Generative AI library not found. (`pip install google-generativeai`)")
    exit()

# --- Configuration Loading & Validation ---
logger.info("Loading configuration...")
# Load secrets from environment
REAMAE_SUBDOMAIN = os.getenv('REAMAE_SUBDOMAIN')
REAMAE_LOGIN_EMAIL = os.getenv('REAMAE_LOGIN_EMAIL')
REAMAE_API_TOKEN = os.getenv('REAMAE_API_TOKEN')
TRIGGER_TAG_NAME = os.getenv('TRIGGER_TAG_NAME')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Validate essential secrets
essential_secrets = {
    "REAMAE_SUBDOMAIN": REAMAE_SUBDOMAIN, "REAMAE_LOGIN_EMAIL": REAMAE_LOGIN_EMAIL,
    "REAMAE_API_TOKEN": REAMAE_API_TOKEN, "TRIGGER_TAG_NAME": TRIGGER_TAG_NAME,
    "GOOGLE_API_KEY": GOOGLE_API_KEY
}
missing_secrets = [k for k, v in essential_secrets.items() if not v]
if missing_secrets:
    logger.critical(f"Missing essential environment variables: {', '.join(missing_secrets)}")
    exit()

API_BASE_URL = f"https://{REAMAE_SUBDOMAIN}.reamaze.com/api/v1"
logger.info(f"API Base URL set to: {API_BASE_URL}")
DATABASE_FILE = 'workflow_state.db' # Using DB now
logger.info(f"Using database file: {DATABASE_FILE}")
PURCHASE_SYSTEM_PATTERN = "placed an order ("
PURCHASE_KEYWORDS = ["payment successful", "order #", "completed my order"]
logger.info("Purchase indicators defined.")

try:
    # Default to production times if env vars not set
    WAIT_1_SECONDS = int(os.getenv('WAIT_1_MINUTES', 3)) * 60
    WAIT_2_SECONDS = int(os.getenv('WAIT_2_MINUTES', 5)) * 60
    logger.info(f"Follow-up Wait times (seconds): Wait1={WAIT_1_SECONDS}, Wait2={WAIT_2_SECONDS}")
except ValueError:
    logger.critical("Invalid number format for WAIT_1_MINUTES or WAIT_2_MINUTES in environment variables.")
    exit()

GEMINI_MODEL = APP_CONFIG.get('gemini_model', 'gemini-1.5-flash-latest')

# --- Initialize Google Generative AI Client ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_generation_config = genai.types.GenerationConfig(
        temperature=APP_CONFIG.get('gemini_summary_temperature', 0.7),
        max_output_tokens=APP_CONFIG.get('gemini_summary_max_output_tokens', 200)
    )
    # Define safety settings (
    gemini_model_instance = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=gemini_generation_config,
        )
    logger.info(f"Google Generative AI client configured for model: {GEMINI_MODEL}")
except Exception as e:
    logger.critical(f"Failed to configure Google Generative AI client: {e}", exc_info=True)
    exit()

# --- Helper Functions ---
logger.debug("Defining helper functions...")

# --- Database Initialization ---
def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    logger.info(f"Initializing database: {DATABASE_FILE}")
    try:
        with sqlite3.connect(DATABASE_FILE, timeout=10) as conn: # Added timeout
            cursor = conn.cursor()
            # Added last_updated column
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    slug TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    added_time TEXT NOT NULL,
                    check1_time TEXT,
                    followup1_sent_time TEXT,
                    check2_time TEXT,
                    last_updated TEXT NOT NULL
                )
            ''')
            # Optional: Add index for status lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON conversations (status)')
            conn.commit()
            logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.critical(f"Database initialization failed: {e}", exc_info=True)
        exit()

# --- API and State Helpers ---
def get_reamaze_auth_header():
    email = os.getenv('REAMAE_LOGIN_EMAIL')
    token = os.getenv('REAMAE_API_TOKEN')
    if not email or not token: logger.error("Missing Reamaze email or token env vars."); return None
    credentials = f"{email}:{token}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    return {'Accept': 'application/json', 'Authorization': f'Basic {encoded_credentials}'}

def make_api_request(method, url, **kwargs):
    headers = get_reamaze_auth_header()
    if not headers: return None
    if 'data' in kwargs or 'json' in kwargs: headers['Content-Type'] = 'application/json'
    for attempt in range(APP_CONFIG.get('max_api_retries', 3)):
        try:
            logger.debug(f"API Request ({method} - Attempt {attempt+1}/{APP_CONFIG.get('max_api_retries', 3)}): URL={url}")
            response = requests.request(method, url, headers=headers, timeout=25, **kwargs)
            logger.debug(f"API Response Status: {response.status_code}")
            # Success or client error we don't retry
            if response.ok or (400 <= response.status_code < 500 and response.status_code not in [429]): # Exclude 429 from non-retriable
                 response.raise_for_status() # Raise HTTPError for non-retriable 4xx, does nothing for 2xx
                 return response
            # Retriable server error or rate limit
            elif response.status_code >= 500 or response.status_code == 429:
                logger.warning(f"Retriable API error ({response.status_code}) on attempt {attempt+1}. Retrying in {APP_CONFIG.get('api_retry_delay_seconds', 5)}s...")
                time.sleep(APP_CONFIG.get('api_retry_delay_seconds', 5))
                # No continue here, loop handles next attempt
            else: # Should technically not be reached if logic is sound
                 logger.error(f"Unexpected HTTP status code {response.status_code} encountered, not retrying.")
                 response.raise_for_status() # Raise for unexpected codes
                 return response # Should not be reached

        except requests.exceptions.Timeout:
            logger.warning(f"API Timeout ({method} - Attempt {attempt+1}) for {url}. Retrying...")
            # Let loop handle sleep/retry
        except requests.exceptions.ConnectionError as ce:
             logger.warning(f"API Connection Error ({method} - Attempt {attempt+1}) for {url}. Retrying...")
             # Let loop handle sleep/retry
        except requests.exceptions.RequestException as e:
            # Catch other non-timeout/connection requests exceptions (like HTTPError raised above)
            logger.error(f"API Request Exception ({method} - Attempt {attempt+1}) for {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 logger.error(f"API Response Text: {e.response.text}")
            return None # Failed, don't retry non-transient request errors

        # Wait before next retry if not the last attempt
        if attempt < MAX_API_RETRIES - 1:
            # Check if we should sleep (i.e., if it wasn't a success or non-retriable error)
             if not ('response' in locals() and (response.ok or (400 <= response.status_code < 500 and response.status_code != 429))):
                time.sleep(API_RETRY_DELAY_SECONDS)
        else:
            logger.error(f"API request failed after {MAX_API_RETRIES} retries for {method} {url}")
            return None # Explicitly return None after all retries fail

    # Fallback if loop somehow finishes without returning (shouldn't happen)
    logger.error(f"Exited retry loop unexpectedly for {method} {url}")
    return None


def get_messages(conversation_slug):
    logging.info(f"Getting messages for slug: {conversation_slug}")
    url = f"{API_BASE_URL}/conversations/{conversation_slug}/messages"
    response = make_api_request('GET', url)
    if response and response.ok:
        try:
            messages = response.json().get('messages', [])
            logging.info(f"Fetched {len(messages)} messages for {conversation_slug}")
            return messages
        except json.JSONDecodeError:
             logging.error(f"Invalid JSON received getting messages for {conversation_slug}")
             return None
    else:
        logging.error(f"Failed to get messages for {conversation_slug} after retries.")
        return None

def check_for_purchase(messages):
    # ... (Keep this function as it was) ...
    logging.debug("Checking for purchase indicators...")
    if not messages: logging.debug("No messages provided."); return False
    logging.debug(f"Checking {len(messages)} messages...")
    for message in messages:
        body = message.get('body', '')
        if isinstance(body, str) and PURCHASE_SYSTEM_PATTERN.lower() in body.lower(): logging.info("Purchase detected (System Pattern)."); return True
    for message in messages:
        body = message.get('body', '')
        if isinstance(body, str):
            body_lower = body.lower()
            for keyword in PURCHASE_KEYWORDS:
                if keyword.lower() in body_lower: logging.info(f"Purchase detected (Keyword: '{keyword}')."); return True
    logging.info("No purchase indicators found.")
    return False

def send_reamaze_message(conversation_slug, message_body):
    # ... (Keep this function, it now uses make_api_request) ...
    logging.info(f"Sending message to slug: {conversation_slug}")
    url = f"{API_BASE_URL}/conversations/{conversation_slug}/messages"
    payload = json.dumps({"message": {"body": message_body}})
    response = make_api_request('POST', url, data=payload)
    if response and response.ok: logging.info(f"Message sent successfully (Status: {response.status_code})"); return True
    else: logging.error(f"Failed to send message to {conversation_slug} after retries."); return False



def format_conversation_for_llm(messages: list) -> str:
    # ... (Keep this function as it was) ...
    logger.debug(f"Formatting {len(messages)} messages for LLM.")
    formatted_text = ""
    if not messages: return ""
    # Consider limiting messages included based on token count or time
    for msg in messages[-15:]: # Example: Limit to last 15 messages
        speaker = "Agent"; user = msg.get('user', {}); body = msg.get('body', '')
        if user and 'email' in user and '@' in user['email'] and not user['email'].endswith(f'@{REAMAE_SUBDOMAIN}.reamaze.com'): speaker = "Customer"
        if body: formatted_text += f"{speaker}: {body}\n"
    logger.debug(f"Formatted text length: {len(formatted_text)}")
    return formatted_text.strip()

def get_gemini_response(prompt_template: str, conversation_text: str) -> str | None:
    # ... (Keep this function as it was, calling Gemini) ...
    logger.info("Getting Gemini response...")
    if not conversation_text: logger.warning("Cannot get Gemini response: Conversation text is empty."); return None
    final_prompt = prompt_template.format(conversation_text=conversation_text)
    try:
        logger.debug(f"Sending prompt to Gemini model {GEMINI_MODEL}")
        response = gemini_model_instance.generate_content(final_prompt)
        if response.candidates and hasattr(response.candidates[0], 'content') and response.candidates[0].content.parts:
            full_response = response.text
            logger.info("Gemini response received successfully.")
            logger.debug(f"Gemini Full Response: {full_response}")
            return full_response.strip()
        else:
            try: logger.warning(f"Gemini generation blocked/empty. Reason: {response.prompt_feedback.block_reason}. Safety: {response.prompt_feedback.safety_ratings}")
            except Exception: logger.warning("Gemini generation blocked or empty. Details unavailable.")
            return None
    except Exception as e: logger.exception(f"Unexpected error getting Gemini response: {e}"); return None

logger.debug("Helper functions defined.")

# --- Core Logic Functions ---
logger.debug("Defining core logic functions...")

def check_for_new_conversations():
    """Polls Re:amaze and adds NEW tagged conversations to the database."""
    logging.info("JOB Starting: check_for_new_conversations")
    conn = None # Initialize conn to None
    try:
        # API Polling part remains the same
        params = { 'tag': TRIGGER_TAG_NAME, 'sort': 'updated_at', 'order': 'desc', 'per_page': 50 }
        url = f"{API_BASE_URL}/conversations"
        response = make_api_request('GET', url, params=params)
        if not response or not response.ok: logging.error("Failed to poll convos."); return
        try: conversations = response.json().get('conversations', [])
        except json.JSONDecodeError: logging.error("Invalid JSON polling convos."); return
        logging.info(f"API Poll found {len(conversations)} convos with tag '{TRIGGER_TAG_NAME}'.")
        if not conversations: logging.info("No tagged convos matching criteria found."); return

        # --- Database Interaction ---
        newly_added_count = 0
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Get existing slugs first
        cursor.execute("SELECT slug FROM conversations")
        existing_slugs = {row[0] for row in cursor.fetchall()}
        logging.debug(f"Found {len(existing_slugs)} slugs already in database.")

        for conv in conversations:
            slug = conv.get('slug')
            if slug and slug not in existing_slugs:
                # Insert new conversation
                logging.info(f"  +++ Found NEW tagged convo: {slug}")
                now_iso = datetime.now(timezone.utc).isoformat()
                try:
                    cursor.execute("""
                        INSERT INTO conversations (slug, status, added_time, last_updated)
                        VALUES (?, ?, ?, ?)
                    """, (slug, 'SCHEDULED_CHECK_1', now_iso, now_iso))
                    newly_added_count += 1
                except sqlite3.IntegrityError:
                    logging.warning(f"Slug {slug} already existed despite initial check (IntegrityError). Skipping insert.")
                except sqlite3.Error as e:
                    logging.error(f"Failed to insert slug {slug}: {e}")

        # Commit all inserts at once after the loop
        if newly_added_count > 0:
            conn.commit()
            logging.info(f"Added {newly_added_count} new conversations to database.")
        else:
            logging.debug("No *new* conversations to add to database.")

    except Exception as e:
        logging.exception("UNEXPECTED ERROR in check_for_new_conversations job")
        # Rollback potential changes if error occurred mid-transaction
        if conn:
            try: # <-- Indentation fixed here
                conn.rollback()
                logging.warning("Rolled back database changes due to error in check_for_new_conversations.")
            except sqlite3.Error as rb_e:
                logging.error(f"Rollback failed: {rb_e}")
    finally:
        # Ensure connection is closed
        if conn:
            try: # <-- Indentation fixed here
                conn.close()
                logging.debug("DB connection closed for check_for_new_conversations.")
            except sqlite3.Error as close_e:
                logging.error(f"Failed to close DB connection: {close_e}")
        logging.info("JOB Finished: check_for_new_conversations")


def process_conversations():
    """Processes conversations based on their state stored in SQLite."""
    logging.info("JOB Starting: process_conversations")
    conn = None
    slugs_processed_in_batch = 0 # Count slugs processed in this run
    try:
        conn = sqlite3.connect(DATABASE_FILE, timeout=10) # Added timeout
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()

        # Define final states to exclude from processing
        final_states_tuple = tuple(['COMPLETED_PURCHASE_DETECTED_1', 'COMPLETED_PURCHASE_DETECTED_2',
                                    'COMPLETED_NO_PURCHASE', 'ERROR_SENDING_1', 'ERROR_SENDING_2',
                                    'ERROR_STATE', 'ERROR_PROCESSING'])
        placeholders = ', '.join('?' * len(final_states_tuple))
        query = f"SELECT * FROM conversations WHERE status NOT IN ({placeholders})"

        cursor.execute(query, final_states_tuple)
        active_conversations = cursor.fetchall()

        if not active_conversations:
            logging.info("No active conversations found in DB to process.")
            return # Exit job early

        logging.info(f"Processing {len(active_conversations)} active slugs from database...")

        for row in active_conversations:
            data = dict(row)
            slug = data['slug']
            status = data.get('status')
            added_time_str = data.get('added_time')
            check1_time_str = data.get('check1_time')
            check2_time_str = data.get('check2_time')

            logging.debug(f"  Checking slug: {slug}, Status: {status}")

            update_needed = False # Flag to track if DB update is needed for this slug
            new_status = status   # Assume status doesn't change unless logic dictates
            update_params = {}  # Store specific fields to update

            try: # Process each slug individually to isolate errors
                # --- Process SCHEDULED_CHECK_1 ---
                if status == 'SCHEDULED_CHECK_1':
                    if not added_time_str:
                        raise ValueError("Missing added_time")

                    if not check1_time_str:
                        added_time = datetime.fromisoformat(added_time_str.replace('Z', '+00:00'))
                        check1_time = added_time + timedelta(seconds=WAIT_1_SECONDS)
                        check1_time_str = check1_time.isoformat()
                        update_params['check1_time'] = check1_time_str
                        logging.info(f"    Calculated Check 1 time for {slug}: {check1_time_str}")
                        update_needed = True
                        # Allow check below to run immediately if needed

                    try:
                        check1_due_time = datetime.fromisoformat(check1_time_str.replace('Z', '+00:00'))
                    except ValueError:
                         logging.error(f"    Invalid check1_time format in DB: {check1_time_str}. Setting to ERROR_STATE.")
                         new_status = 'ERROR_STATE'
                         update_needed = True
                    else:
                        if now_utc >= check1_due_time:
                            logging.info(f"    >>> Performing Check 1 for {slug} (due at {check1_due_time})...")
                            messages = get_messages(slug)
                            if messages is None:
                                logging.warning(f"    Failed get messages Check 1 ({slug}). Will retry next cycle.")
                            else:
                                if check_for_purchase(messages):
                                    logging.info(f"      Purchase detected Check 1 ({slug}). Complete.")
                                    new_status = 'COMPLETED_PURCHASE_DETECTED_1'
                                    # remove_reamaze_tag(slug, TRIGGER_TAG_NAME) # <-- TAG REMOVAL COMMENTED OUT
                                else:
                                    logging.info(f"      No purchase Check 1. Generate & Send F1 ({slug}).")
                                    formatted_conv = format_conversation_for_llm(messages)
                                    prompt1_template = APP_CONFIG.get('llm_prompt_followup1', "Error: Prompt 1 missing.")
                                    llm_response_text = get_gemini_response(prompt_template=prompt1_template, conversation_text=formatted_conv)

                                    if llm_response_text:
                                        followup1_text = llm_response_text
                                        logging.debug(f"      Constructed F1 using Gemini response.")
                                    else:
                                        logging.warning(f"      Gemini response failed for {slug}. Using fallback.")
                                        followup1_text = APP_CONFIG.get('followup1_fallback_message', "Error: Fallback missing.")

                                    if send_reamaze_message(slug, followup1_text):
                                        new_status = 'SCHEDULED_CHECK_2'
                                        update_params['followup1_sent_time'] = now_iso
                                        check2_time = now_utc + timedelta(seconds=WAIT_2_SECONDS)
                                        update_params['check2_time'] = check2_time.isoformat()
                                        logging.info(f"      Sent F1. Scheduling Check 2 at {update_params['check2_time']}")
                                    else:
                                        new_status = 'ERROR_SENDING_1'
                                        logging.error(f"      Failed to send F1 for {slug} after retries.")
                                update_needed = True
                        else:
                             logging.debug(f"    Check 1 for {slug} not due yet (due at {check1_due_time}).")

                # --- Process SCHEDULED_CHECK_2 ---
                elif status == 'SCHEDULED_CHECK_2':
                    if not check2_time_str:
                        raise ValueError("Missing check2_time")

                    try:
                        check2_due_time = datetime.fromisoformat(check2_time_str.replace('Z', '+00:00'))
                    except ValueError:
                        logging.error(f"    Invalid check2_time format in DB: {check2_time_str}. Setting to ERROR_STATE.")
                        new_status = 'ERROR_STATE'
                        update_needed = True
                    else:
                        if now_utc >= check2_due_time:
                            logging.info(f"    >>> Performing Check 2 for {slug} (due at {check2_due_time})...")
                            messages = get_messages(slug)
                            if messages is None:
                                logging.warning(f"    Failed get messages Check 2 ({slug}). Will retry next cycle.")
                            else:
                                if check_for_purchase(messages):
                                    logging.info(f"      Purchase detected Check 2 ({slug}). Complete.")
                                    new_status = 'COMPLETED_PURCHASE_DETECTED_2'
                                    # remove_reamaze_tag(slug, TRIGGER_TAG_NAME) # <-- TAG REMOVAL COMMENTED OUT
                                else:
                                    logging.info(f"      No purchase Check 2. Generate & Send F2 ({slug}).")
                                    formatted_conv = format_conversation_for_llm(messages)
                                    prompt2_template = APP_CONFIG.get('llm_prompt_followup2', "Error: Prompt 2 missing.")
                                    llm_response_text_f2 = get_gemini_response(prompt_template=prompt2_template, conversation_text=formatted_conv)

                                    if llm_response_text_f2:
                                        followup2_text = llm_response_text_f2
                                        logging.debug(f"      Constructed F2 using Gemini response.")
                                    else:
                                        logging.warning(f"      Gemini F2 failed for {slug}. Using fallback.")
                                        followup2_text = APP_CONFIG.get('followup2_fallback_message', "Error: Fallback F2 missing.")

                                    if send_reamaze_message(slug, followup2_text):
                                        new_status = 'COMPLETED_NO_PURCHASE'
                                        logging.info(f"      Successfully sent F2 for {slug}.")
                                        # remove_reamaze_tag(slug, TRIGGER_TAG_NAME) # <-- TAG REMOVAL COMMENTED OUT
                                    else:
                                        new_status = 'ERROR_SENDING_2'
                                        logging.error(f"      Failed to send F2 for {slug} after retries.")
                                update_needed = True
                        else:
                             logging.debug(f"    Check 2 for {slug} not due yet (due at {check2_due_time}).")

            except ValueError as ve:
                logging.error(f"  Data Error processing slug {slug}: {ve}. Setting to ERROR_STATE.")
                new_status = 'ERROR_STATE'
                update_needed = True
            except Exception as inner_e:
                 logging.exception(f"  UNEXPECTED ERROR processing slug {slug}: {inner_e}")
                 new_status = 'ERROR_PROCESSING'
                 update_needed = True

            if update_needed:
                try:
                    update_params['status'] = new_status
                    update_params['last_updated'] = now_iso
                    set_clause = ", ".join([f"{key} = ?" for key in update_params])
                    sql_update = f"UPDATE conversations SET {set_clause} WHERE slug = ?"
                    update_values = list(update_params.values()) + [slug]
                    logging.debug(f"    Executing DB Update for {slug}: SET {update_params}")
                    cursor.execute(sql_update, update_values)
                    slugs_processed_in_batch += 1
                except sqlite3.Error as db_e:
                     logging.error(f"  Failed to UPDATE database for slug {slug}: {db_e}")

        if slugs_processed_in_batch > 0:
             try:
                 conn.commit()
                 logging.info(f"Committed updates potentially affecting up to {slugs_processed_in_batch} slugs.")
             except sqlite3.Error as commit_e:
                 logging.error(f"DATABASE COMMIT FAILED: {commit_e}. Changes for this batch may be lost.")
                 try: conn.rollback(); logging.warning("Rolled back changes after commit failure.")
                 except sqlite3.Error as rb_e: logging.error(f"Rollback after commit failure also failed: {rb_e}")
        else:
             logging.debug("No database updates were needed or successfully prepared in this processing job.")

    except sqlite3.Error as e:
        logging.error(f"Database error during process_conversations job: {e}", exc_info=True)
        if conn:
            try: conn.rollback(); logging.warning("Rolled back changes due to top-level DB error.")
            except sqlite3.Error as rb_e: logging.error(f"Rollback failed: {rb_e}")
    except Exception as e:
        logging.exception("UNEXPECTED Top-Level ERROR in process_conversations job")
    finally:
        if conn:
            try: conn.close(); logging.debug("DB connection closed for process_conversations.")
            except sqlite3.Error as close_e: logging.error(f"Failed to close DB conn: {close_e}")
        logging.info("JOB Finished: process_conversations")
# --- Main execution block ---
if __name__ == "__main__":
    logger.info("Entering main execution block.")
    try:
        init_db()      # Initialize the database
        load_config()  # Load external config file

        logger.info("Setting up scheduler...")
        scheduler = BlockingScheduler(timezone=timezone.utc)
        logger.info("Scheduler initialized.")

        poll_interval_mins = APP_CONFIG.get('poll_interval_minutes', 15)
        process_interval_mins = APP_CONFIG.get('process_interval_minutes', 5)
        logger.info(f"Scheduler Intervals: Poll={poll_interval_mins} mins, Process={process_interval_mins} mins")

        first_poll_run_time = datetime.now(timezone.utc) + timedelta(seconds=5)
        first_process_run_time = datetime.now(timezone.utc) + timedelta(seconds=10)

        logger.info(f"Adding 'check_for_new_conversations' job. Interval: {poll_interval_mins} mins.")
        scheduler.add_job( check_for_new_conversations, 'interval', minutes=poll_interval_mins,
            next_run_time=first_poll_run_time, id='poll_reamaze_job', replace_existing=True )

        logger.info(f"Adding 'process_conversations' job. Interval: {process_interval_mins} mins.")
        scheduler.add_job( process_conversations, 'interval', minutes=process_interval_mins,
            next_run_time=first_process_run_time, id='process_state_job', replace_existing=True )

        logger.info("-" * 40 + "\n--- SCHEDULER SETUP COMPLETE ---")
        logger.info(f"{datetime.now(timezone.utc)} - Scheduler started. Press Ctrl+C to exit.\n" + "-" * 40)

        scheduler.start() # Blocking call

    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received. Shutting down scheduler...")
        if 'scheduler' in locals() and scheduler.running: scheduler.shutdown()
        logger.info("Scheduler shut down gracefully.")
    except Exception as e:
        logger.critical(f"\n{'='*60}\nFATAL ERROR during scheduler setup or runtime: {e}")
        logger.exception("Traceback:")
        logger.critical("="*60)
    finally:
        logger.info("Script exiting.")
else:
    logger.info("Script is being imported, not run directly.")
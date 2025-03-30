import time
from tqdm import tqdm
import tweepy
from twitter_auth import TwitterAuthenticator
import pandas as pd
from tweepy import API, Client, OAuth1UserHandler
import keys
class Scraper():
    def __init__(self, client):
        self.client: API | Client = client
    def get_users(self, uids: list, **kwargs) -> pd.DataFrame:
        users = self.client.get_users(ids=uids, **kwargs)
        return [user.data for user in users.data]

    def get_user(self, uids: int | str | None, username: str | None = None, **kwargs) -> dict:
        return self.client.get_user(id = uids,username=username, **kwargs).data.data
    def lookup_users(self, uids: list[int]= None, usernames : list[str] = None, **kwargs)-> pd.DataFrame:
        users = self.client.lookup_users(user_id=uids, screen_name=usernames, **kwargs)
        return [vars(user.data) for user in users.data]
    
def bot_user_ids():
    users = pd.read_csv("data/raw/label.csv")
    users["id"] = users["id"].str.removeprefix("u").astype(int)
    return users["id"][users['label']== "bot"]

def create_client():
    auth = OAuth1UserHandler(
        consumer_key=keys.CONSUMER_KEY,
        consumer_secret=keys.CONSUMER_SECRET,
        access_token= keys.ACCESS_TOKEN,
        access_token_secret=keys.ACCESS_TOKEN_SECRET)
    return TwitterAuthenticator(auth=auth).get_client()

def scrape(client, users: list = None,  choice: int = None, fields: dict = None):
    if not users:
        print("No users.")
        return
    scraper = Scraper(client)
    scrape_method = scraper.get_user
    if choice == 2:
        if isinstance(client, Client):
            scrape_method = scraper.get_users
        if isinstance(client, API):
            scrape_method = scraper.lookup_users
    return scrape_method(uids=users, user_fields = fields["user"], tweet_fields=fields["tweet"])

def chunker(data:list, num_lenght:int)-> list:
    chunks = []
    for i in range(0,len(data),num_lenght):
        chunks.append(data[i:i+num_lenght])
    return chunks

def to_csv( filename: str, data: list[dict]| pd.DataFrame = None):
    if isinstance(data, list[dict]):
        data = pd.DataFrame(data)
    if 'new_col' in data.columns1:
        data.insert(0, 'new_col', data.pop('new_col'))
    data.to_csv(f"data/raw/{filename}.csv", index=False)

def process_chunks(client, chunks, choice, fields):
    data = []
    pbar = tqdm(chunks, desc="Scraping chunks")
    for chunk in pbar:
        attempt = 0
        while attempt < 2:
            try:
                data.extend(scrape(users=chunk, client=client, choice=choice, fields=fields))
                pbar.update(1)
                break  # Exit retry loop on success
            except tweepy.TooManyRequests as e:
                if attempt == 0:
                    time.sleep(1)  # Retry after delay
                else:
                    pbar.write(f"Failed chunk: {chunk} | Error: {e}")
            attempt += 1
    return data

def make_request():
    user_fields = [
    "id",                        # Unique identifier (default)
    "name",                      # Profile name (default)
    "username",                  # Handle/screen name (default)
    "affiliation",               # Organizational affiliation (e.g., government, business)
    "connection_status",         # Relationship status (following, blocking, etc.)
    "created_at",                # Account creation date (ISO 8601)
    "description",               # Bio text
    "entities",                  # Hashtags, URLs, mentions in bio
    "is_identity_verified",      # Additional identity verification (beyond blue check)
    "location",                  # Freeform location string
    "most_recent_tweet_id",      # ID of the user's most recent Tweet
    "parody",                    # Whether the account is a parody
    "pinned_tweet_id",           # ID of pinned Tweet
    "profile_banner_url",        # URL of profile header/banner image
    "profile_image_url",         # URL of profile image
    "protected",                 # Are Tweets private?
    "public_metrics",            # Follower/following/tweet counts
    "receives_your_dm",          # Whether the user can receive your DMs
    "subscription",              # Paid subscription status (X Premium)
    "subscription_type",         # Type of subscription (e.g., "blue", "gold")
    "url",                       # Profile URL
    "verified",                  # Is the account verified?
    "verified_followers_count",  # Number of verified followers
    "verified_type",             # Type of verification (e.g., "government", "business")
    "withheld"                   # Content withholding details (legal/tos reasons)
    ]
    tweet_fields = [
    # Default fields (always included)
    "id",                   # string - Unique identifier of the Tweet
    "text",                 # string - UTF-8 text content of the Tweet
    "edit_history_tweet_ids", # object - IDs of all Tweet versions (edit history)

    # Optional fields
    "attachments",          # object - Media/attachment details (requires expansions)
    "author_id",            # string - User ID of the Tweet author
    "context_annotations",  # array - Topic/entity annotations
    "conversation_id",      # string - ID of the root Tweet in a thread
    "created_at",           # ISO 8601 date - Tweet creation time
    "edit_controls",        # object - Edit eligibility and remaining edits
    "entities",             # object - Parsed hashtags, URLs, mentions
    "in_reply_to_user_id",  # string - Original Tweet's author ID (for replies)
    "lang",                 # string - Detected language of the Tweet
    "possibly_sensitive",   # boolean - Sensitivity flag for content
    "public_metrics",       # object - Public engagement metrics
    "referenced_tweets",    # array - Related Tweets (retweets/quotes/replies)
    "reply_settings",       # string - Who can reply ("everyone"/"mentioned_users"/"followers")
    "withheld",             # object - Content withholding details

    # Metrics requiring OAuth 2.0 user context
    "non_public_metrics",   # object - Private impressions metrics
    "organic_metrics",      # object - Organic engagement metrics
    "promoted_metrics"      # object - Promoted engagement metrics
    ]
    fields = {
        "user": user_fields,
        "tweet": tweet_fields
        }
    client = create_client()
    users = bot_user_ids().sample(n=10000,random_state=21).to_list()

    choice = input("Choose: \n1. Single User \n2. Multiple \nInput: ")
    choice = int(choice) if choice in {"1", "2"} else 2

    chunks = chunker(users, 1)[0] if choice == 1 else chunker(users, 10)
    data = process_chunks(client=client, chunks=chunks, choice=choice, fields=fields)
    to_csv("x-data_10k", data=data)

if __name__ == "__main__":
    make_request()
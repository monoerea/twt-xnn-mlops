import tweepy
from twitter_auth import TwitterAuthenticator
import pandas as pd
from tweepy import API, Client, OAuth1UserHandler
import keys
class Scraper():
    def __init__(self, client):
        self.client: API | Client = client
    def get_users(self, uids: list, **kwargs) -> pd.DataFrame:
        user = self.client.get_users(ids=uids, **kwargs)
        print(user)
        for i in user:
            try:
                print(i._json)
            except Exception as e:
                print(e)
        users = []
        for obj in user.data:
            users.append(vars(obj))
        print("Same size:", len(users)==len(users.data))
        return pd.DataFrame(users)

    def get_user(self, uid: int = None, username: str = None, **kwargs) ->dict:
        user = self.client.get_user(uid = uid, username=username,  **kwargs)
        return vars(user.data)

    def lookup_users(self, uids: list[int]= None, usernames : list[str] = None, **kwargs)-> pd.DataFrame:
        """Return up to 100 fully hydrated user objects in dataframe

        Args:
            uids (list[int]): List of up to 100 X ids
            usernames (list[str]): List of up to 100 X usernames

        Returns:
            pd.DataFrame: For saving to csv file or data analysis
        """
        if isinstance(self.client, API):
            raise TypeError("self.client must be an instance of Client")
        users = self.client.lookup_users(user_id=uids, screen_name=usernames, **kwargs)
        user_list = []
        for obj in users.data:
            user_list.append(vars(obj))
        df = pd.DataFrame(user_list)
        return df
def bot_user_ids():
    users = pd.read_csv("data/raw/label.csv")
    users.head(10)
    users["id"] = users["id"].str.removeprefix("u").astype(int)
    return users["id"][users['label']== "bot"]
def scrape(users: list = None):
    auth = OAuth1UserHandler(
        consumer_key=keys.CONSUMER_KEY,
        consumer_secret=keys.CONSUMER_SECRET,
        access_token='1569207846893670400-BZKMpsnmkaqCWUYtVgGjeU6IPQa1qY',access_token_secret='JMOnXtVkAAn2xRuHyKuK3HFdaKoUZcr25IK0HJg2LrHMn')
    xauth = TwitterAuthenticator(auth=auth)
    client = xauth.get_client()
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
    if isinstance(client, Client):
        scraper = Scraper(client)
        #scraper.get_users(uids = users)
        user = scraper.get_users(uids=users, user_fields = user_fields,)
        df = pd.DataFrame(user)
        #df: pd.DataFrame = scraper.get_user(uids=users)
        df.to_csv("data/raw/user_100.csv")
import random
import time

def make_request():
    try:
        users = bot_user_ids().tolist()[:100]
        scrape(users=users)
    except tweepy.TooManyRequests as e:
        print(e)
        return make_request()
def main():
    make_request()
if __name__ == "__main__":
    main()
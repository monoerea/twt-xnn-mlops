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
            print(i._json)
        return user


def bot_user_ids():
    users = pd.read_csv("data/raw/label.csv")
    users.head(10)
    users["id"] = users["id"].str.removeprefix("u").astype(int)
    return users["id"][users['label']== "bot"]
def scrape(users):
    auth = OAuth1UserHandler(
        consumer_key=keys.CONSUMER_KEY,
        consumer_secret=keys.CONSUMER_SECRET,
        access_token='1569207846893670400-BZKMpsnmkaqCWUYtVgGjeU6IPQa1qY',access_token_secret='JMOnXtVkAAn2xRuHyKuK3HFdaKoUZcr25IK0HJg2LrHMn')
    xauth = TwitterAuthenticator(auth=auth)
    client = xauth.get_client()
    scraper = Scraper(client)
    scraper.get_users(uids = users)
def main():
    users = bot_user_ids().tolist()[:100]
    scrape(users=users)
if __name__ == "__main__":
    main()
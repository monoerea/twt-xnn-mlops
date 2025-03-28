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

def main():
    users = [1464599533,1435309448,19932466,940132086,1945343251,2772953185,2683063328,513001579,162572003,826306403,16641565,2756873076,486160284,18746944,2323141220,2793171236,18623405,171848975,2549103608,36790442,2835488103,1043436650,2837972170]
    auth = OAuth1UserHandler(
        consumer_key=keys.CONSUMER_KEY,
        consumer_secret=keys.CONSUMER_SECRET,
        access_token='1569207846893670400-BZKMpsnmkaqCWUYtVgGjeU6IPQa1qY',access_token_secret='JMOnXtVkAAn2xRuHyKuK3HFdaKoUZcr25IK0HJg2LrHMn')
    xauth = TwitterAuthenticator(auth=auth)
    client = xauth.get_client()
    scraper = Scraper(client)
    scraper.get_users(uids = users)
if __name__ == "__main__":
    main()
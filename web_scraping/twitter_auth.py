import keys
from tweepy import OAuth1UserHandler, API, Client, TweepyException

class TwitterAuthenticator():
    def __init__(self, auth: OAuth1UserHandler = None, client: Client | API = None):
        self.auth = auth
        self.client = client

    def is_verified(self):
        """Checks if the user has valid authentication tokens."""
        if isinstance(self.client, Client):
            return all([self.client.access_token, self.client.access_token_secret])
        return self.auth and all([self.auth.access_token, self.auth.access_token_secret])

    def get_client(self)-> Client | None:
        if not self.is_verified():
            raise ValueError("Client is not authenticated. Call `verify` first.")
        if isinstance(self.client, Client):
            return self.client
        if not self.auth:
            raise ValueError("No authentication handler (auth) provided.")
        try:
            self.client = Client(
            bearer_token=keys.BEARER_TOKEN,
            consumer_key=keys.CONSUMER_KEY,
            consumer_secret=keys.CONSUMER_SECRET,
            access_token=self.auth.access_token,
            access_token_secret=self.auth.access_token_secret,
            wait_on_rate_limit=True
            )
            return self.client
        except TweepyException  as e:
            print(f"Error creating Client: {e}")

    def get_api(self) -> API | None:
        if self.is_verified() and self.client:
            return self.client
        if isinstance(self.client, API):
            return self.client
        if not self.auth:
            raise ValueError("No authentication handler (auth) provided.")
        try:
            self.client = API(auth=self.auth,                 wait_on_rate_limit=True)
            return self.client
        except TweepyException  as e:
            print(f"Error creating Client: {e}")
    def get_auth_url(self):
        if self.is_verified() == True:
            return
        if not self.auth:
            self.auth = OAuth1UserHandler(
                consumer_key=keys.CONSUMER_KEY,
                consumer_secret=keys.CONSUMER_SECRET,
                callback="oob"
            )
        return self.auth.get_authorization_url(signin_with_twitter=True)
    def verify(self, pin):
        if self.is_verified() == True:
            return
        if not self.auth:
            self.auth = OAuth1UserHandler(
                consumer_key=keys.CONSUMER_KEY,
                consumer_secret=keys.CONSUMER_SECRET,
            )
        if pin:
            try:
                access_token, access_token_secret = self.auth.get_access_token(verifier=pin)
                self.auth.set_access_token(access_token, access_token_secret)
            except TweepyException as e:
                print(f"Error during verification: {e}")
        else:
            if not (self.auth.access_token and self.auth.access_token_secret):
                print("Error: PIN is required for verification.")
    def get_tokens(self) -> dict[str, str] | None:
        """Return stored access token and secret after verification."""
        if self.is_verified():
            return {
                "access_token": self.auth.access_token,
                "access_token_secret": self.auth.access_token_secret
            }
        return None


def test_tokens():
    auth = OAuth1UserHandler(
        consumer_key=keys.CONSUMER_KEY,
        consumer_secret=keys.CONSUMER_SECRET,
        access_token='1569207846893670400-BZKMpsnmkaqCWUYtVgGjeU6IPQa1qY',access_token_secret='JMOnXtVkAAn2xRuHyKuK3HFdaKoUZcr25IK0HJg2LrHMn')
    tauth = TwitterAuthenticator(auth=auth)
    api = tauth.get_api()
    tweet = api.get_user(user_id =2244994945)
    print(tweet._json)
def test_main():
    auth = TwitterAuthenticator()
    auth_url = auth.get_auth_url()
    print(f"Please visit this URL to authorize the app: {auth_url}")
    pin = input("Enter the PIN provided by Twitter: ")
    auth.verify(pin)
    auth_tokens = auth.get_tokens()
    if auth_tokens:
        print("Access token and secret retrieved successfully.", auth_tokens)
    else:
        print("Failed to retrieve access token and secret.")
    client = auth.get_client()
    tweet = client.get_user(id = '1569207846893670400')
    print(tweet)
    api = auth.get_api()
    user = api.get_user(user_id = '1569207846893670400')
    print(user._json)
    if client:
        print("Client authenticated successfully.")
    if api:
        print("API authenticated successfully.")
def main():
    #test_tokens()
    test_main()
if __name__ == "__main__":
    main()
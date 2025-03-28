from tweepy import OAuth1UserHandler, API, Client, TweepyException


class TwitterAuthenticator():
    def __init__(self, auth: OAuth1UserHandler = None, client: Client | API = None):
        self.auth = auth
        self.client = client

    def is_verified(self):
        if self.client and hasattr(self.client, '_get_oauth_1_authenticating_user_id'):
            return self.client._get_oauth_1_authenticating_user_id() is not None
        return self.auth and self.auth.access_token is not None

    def get_client(self)-> Client | None:
        if not self.is_verified():
            raise ValueError("Client is not authenticated. Call `verify` first.")
        if isinstance(self.client, Client):
            return self.client
        if not self.auth:
            raise ValueError("No authentication handler (auth) provided.")
        try:
            self.client = Client(access_token=self.auth.ACCESS_TOKEN, access_token_secret=self.auth.ACCESS_TOKEN_SECRET)
            return self.client
        except TweepyException  as e:
            print(f"Error creating Client: {e}")

    def get_api(self) -> API | None:
        if self.is_verified():
            return self.client
        if isinstance(self.client, API):
            return self.client
        if not self.auth:
            raise ValueError("No authentication handler (auth) provided.")
        try:
            self.client = API(auth=self.auth)
            return self.client
        except TweepyException  as e:
            print(f"Error creating Client: {e}")
    def get_auth_url(self):
        if self.is_verified() == True:
            return
        if not self.auth:
            self.auth = OAuth1UserHandler(
                consumer_key=keys.CONSUMER_KEY,
                consumer_secret=keys.CONSUMER_SECRET
            )
        return self.auth.get_authorization_url()
    def verify(self, pin):
        if self.is_verified() == True:
            return
        if not self.auth:
            self.auth = OAuth1UserHandler(
                consumer_key=keys.CONSUMER_KEY,
                consumer_secret=keys.CONSUMER_SECRET
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


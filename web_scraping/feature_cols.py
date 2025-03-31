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
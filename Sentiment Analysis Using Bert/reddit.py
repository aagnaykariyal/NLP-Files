class RedditClass:
    def __init__(self):
        self.text = "Reddit Initialization"

    @staticmethod
    def redditapi():
        import json
        import praw  # Importing necessary libraries

        # --------- Reddit API Cred --------- #
        reddit = praw.Reddit(
            client_id="zw8uC_emsN_GTyOai4v6OQ",
            client_secret="XJf-l9V2TNGaF9fn6486C0OOEpRplA",
            user_agent="ua"
        )
        # ----------------------------------- #

        subreddits = ["turntables", "vinyl", "audiophile"]

        corpus = []  # List to store the data

        # ---------- Fetching the data ---------- #
        for sreddit in subreddits:
            # Change the value of limit to change the number of posts from the hot section
            for submissions in reddit.subreddit(sreddit).hot(limit=10):
                corpus.append(submissions.title)  # Adding post titles to the list
                post = reddit.submission(id=submissions.id)
                post.comments.replace_more(limit=None)
                for com in post.comments.list():
                    corpus.append(com.body)  # Adding comments to the list

        with open('data.json', "w", newline='') as json_file:
            json.dump({'Data': corpus}, json_file)

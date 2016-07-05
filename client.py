# third-party HTTP client library
import requests

# assume that "app" below is your flask app, and that
# "Response" is imported from flask.

try:
    r = requests.get("http://127.0.0.1:8000/")

    status= r.status_code
    print status
except:
    print "server is down"


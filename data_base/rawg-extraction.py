import os, json, datetime, time, urllib.parse, urllib.request, boto3
from botocore.exceptions import ClientError
from zoneinfo import ZoneInfo

REGION = os.environ.get("REGION", "eu-north-1")

def get_rawg_api_key(secret_id: str) -> str:
    secrets_manager = boto3.client("secretsmanager", region_name=REGION)
    secret_response = secrets_manager.get_secret_value(SecretId=secret_id)
    secret_data = json.loads(secret_response["SecretString"])
    if "RAWG_API_KEY" not in secret_data or not secret_data["RAWG_API_KEY"]:
        raise RuntimeError("El secreto no tiene 'RAWG_API_KEY'")
    return secret_data["RAWG_API_KEY"]

def http_get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "lambda-rawg/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))

def save_to_s3(s3, bucket_name: str, object_key: str, payload: dict):
    s3.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json"
    )

def build_initial_url(base_url: str, api_key: str, page_size: int) -> str:
    params = {"key": api_key, "page_size": page_size, "ordering": "-added"}
    return base_url + "?" + urllib.parse.urlencode(params)

def build_incremental_url(base_url: str, api_key: str, page_size: int, days_back: int) -> str:
    today = datetime.date.today()
    since = today - datetime.timedelta(days=days_back)
    params = {
        "key": api_key,
        "page_size": page_size,
        "dates": f"{since.isoformat()},{today.isoformat()}",
        "ordering": "-updated"
    }
    return base_url + "?" + urllib.parse.urlencode(params)

def lambda_handler(event, context):
    bucket_name = os.environ["BUCKET_NAME"]
    secret_id = os.environ["SECRET_ID"]
    s3_prefix = os.environ.get("S3_PREFIX", "rawg")
    mode = os.environ.get("MODE", "initial").lower()
    page_size = int(os.environ.get("PAGE_SIZE", "40"))
    max_pages = int(os.environ.get("MAX_PAGES", "200"))
    days_back = int(os.environ.get("DAYS_BACK", "1"))

    api_key = get_rawg_api_key(secret_id)

    base_url = "https://api.rawg.io/api/games"
    if mode == "incremental":
        url = build_incremental_url(base_url, api_key, page_size, days_back)
    else:
        url = build_initial_url(base_url, api_key, page_size)

    s3 = boto3.client("s3")
    madrid_timezone = ZoneInfo("Europe/Madrid")
    current_date = datetime.datetime.now(madrid_timezone).date().isoformat()
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    pages_fetched = 0
    saved_files = []

    while url and pages_fetched < max_pages:
        data = http_get_json(url)
        results = data.get("results", [])
        next_url = data.get("next")

        page_number = pages_fetched + 1
        object_key = f"{s3_prefix}/{current_date}/page-{page_number:04d}_{run_id}.json"


        save_to_s3(s3, bucket_name, object_key, data)
        saved_files.append(object_key)

        pages_fetched += 1
        url = next_url
        time.sleep(0.50)

    return {
        "status": "ok",
        "mode": mode,
        "pages_fetched": pages_fetched,
        "saved_examples": saved_files[:3] + (["..."] if len(saved_files) > 3 else [])
    }

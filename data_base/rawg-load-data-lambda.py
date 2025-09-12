import os, json, datetime, boto3
import pg8000

region       = os.environ.get("REGION", "eu-north-1")
bucket_name  = os.environ["S3_BUCKET"]
s3_prefix    = os.environ.get("S3_PREFIX", "rawg")
days_back    = int(os.environ.get("DAYS_BACK", "2"))
max_files    = int(os.environ.get("MAX_FILES", "20"))

db_host      = os.environ["RDS_HOST"]
db_name      = os.environ["RDS_DB"]
db_user      = os.environ["RDS_USER"]
db_password  = os.environ["RDS_PASSWORD"]

s3 = boto3.client("s3", region_name=region)

def get_recent_files():
    files = []
    today = datetime.date.today()
    for d in range(days_back + 1):
        day = today - datetime.timedelta(days=d)
        prefix = f"{s3_prefix}/{day.isoformat()}/"
        resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".json"):
                files.append(key)
            if len(files) >= max_files:
                return files
    return files

def load_json_from_s3(key):
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return obj["Body"].read().decode("utf-8")

def get_connection():
    return pg8000.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=5432,
        ssl_context=True,
    )


# Dimensiones
SQL_DIM_ESRB = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
)
INSERT INTO public.esrb_ratings (esrb_rating_id, name, slug)
SELECT DISTINCT
  (g->'esrb_rating'->>'id')::int,
  g->'esrb_rating'->>'name',
  g->'esrb_rating'->>'slug'
FROM games
WHERE g ? 'esrb_rating'
  AND COALESCE(g->'esrb_rating'->>'id','') <> ''
  AND (g->'esrb_rating'->>'id') ~ '^[0-9]+$'
ON CONFLICT (esrb_rating_id) DO UPDATE
SET name = EXCLUDED.name, slug = EXCLUDED.slug;
"""

SQL_DIM_PLATFORMS = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
),
plats AS (
  SELECT DISTINCT
    (p->'platform'->>'id')::int AS platform_id,
    p->'platform'->>'name'      AS name,
    p->'platform'->>'slug'      AS slug
  FROM games
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(g->'platforms') = 'array' THEN g->'platforms' ELSE '[]'::jsonb END
  ) AS p
)
INSERT INTO public.platforms (platform_id, name, slug)
SELECT platform_id, name, slug FROM plats
ON CONFLICT (platform_id) DO UPDATE
SET name = EXCLUDED.name, slug = EXCLUDED.slug;
"""

SQL_DIM_GENRES = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
),
gs AS (
  SELECT DISTINCT
    (x->>'id')::int AS genre_id,
    x->>'name'      AS name,
    x->>'slug'      AS slug
  FROM games
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(g->'genres') = 'array' THEN g->'genres' ELSE '[]'::jsonb END
  ) AS x
)
INSERT INTO public.genres (genre_id, name, slug)
SELECT genre_id, name, slug FROM gs
ON CONFLICT (genre_id) DO UPDATE
SET name = EXCLUDED.name, slug = EXCLUDED.slug;
"""

SQL_DIM_STORES = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
),
st AS (
  SELECT DISTINCT
    (s->'store'->>'id')::int AS store_id,
    s->'store'->>'name'      AS name,
    s->'store'->>'slug'      AS slug,
    s->'store'->>'domain'    AS domain
  FROM games
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(g->'stores') = 'array' THEN g->'stores' ELSE '[]'::jsonb END
  ) AS s
)
INSERT INTO public.stores (store_id, name, slug, domain)
SELECT store_id, name, slug, domain FROM st
ON CONFLICT (store_id) DO UPDATE
SET name = EXCLUDED.name,
    slug = EXCLUDED.slug,
    domain = COALESCE(EXCLUDED.domain, public.stores.domain);
"""

SQL_DIM_TAGS = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
),
tg AS (
  SELECT DISTINCT
    (t->>'id')::int AS tag_id,
    t->>'name'      AS name,
    t->>'slug'      AS slug
  FROM games
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(g->'tags') = 'array' THEN g->'tags' ELSE '[]'::jsonb END
  ) AS t
)
INSERT INTO public.tags (tag_id, name, slug)
SELECT tag_id, name, slug FROM tg
ON CONFLICT (tag_id) DO UPDATE
SET name = EXCLUDED.name, slug = EXCLUDED.slug;
"""

# Tabla principal
SQL_GAMES = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
)
INSERT INTO public.games (
  game_id, slug, name, released, tba, rating, rating_top, ratings_count,
  reviews_text_count, metacritic, playtime, suggestions_count, updated,
  background_image, added, reviews_count, dominant_color, saturated_color,
  esrb_rating_id, website
)
SELECT
  (g->>'id')::int,
  g->>'slug',
  g->>'name',
  NULLIF(g->>'released','')::date,
  (g->>'tba')::boolean,
  NULLIF(g->>'rating','')::numeric,
  NULLIF(g->>'rating_top','')::int,
  NULLIF(g->>'ratings_count','')::int,
  NULLIF(g->>'reviews_text_count','')::int,
  NULLIF(g->>'metacritic','')::int,
  NULLIF(g->>'playtime','')::int,
  NULLIF(g->>'suggestions_count','')::int,
  NULLIF(g->>'updated','')::timestamp,
  g->>'background_image',
  NULLIF(g->>'added','')::int,
  NULLIF(g->>'reviews_count','')::int,
  g->>'dominant_color',
  g->>'saturated_color',
  NULLIF(g->'esrb_rating'->>'id','')::int,
  g->>'website'
FROM games
ON CONFLICT (game_id) DO UPDATE SET
  slug               = EXCLUDED.slug,
  name               = EXCLUDED.name,
  released           = EXCLUDED.released,
  tba                = EXCLUDED.tba,
  rating             = EXCLUDED.rating,
  rating_top         = EXCLUDED.rating_top,
  ratings_count      = EXCLUDED.ratings_count,
  reviews_text_count = EXCLUDED.reviews_text_count,
  metacritic         = EXCLUDED.metacritic,
  playtime           = EXCLUDED.playtime,
  suggestions_count  = EXCLUDED.suggestions_count,
  updated            = EXCLUDED.updated,
  background_image   = EXCLUDED.background_image,
  added              = EXCLUDED.added,
  reviews_count      = EXCLUDED.reviews_count,
  dominant_color     = EXCLUDED.dominant_color,
  saturated_color    = EXCLUDED.saturated_color,
  esrb_rating_id     = EXCLUDED.esrb_rating_id,
  website            = EXCLUDED.website;
"""

# Tablas puentes N:M
SQL_GAME_PLATFORMS = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
)
INSERT INTO public.game_platforms
  (game_id, platform_id, released_at, requirements_min, requirements_rec)
SELECT
  (g->>'id')::int,
  (p->'platform'->>'id')::int,
  NULLIF(p->>'released_at','')::date,
  COALESCE(p->'requirements'->>'minimum',    p->'requirements_en'->>'minimum'),
  COALESCE(p->'requirements'->>'recommended', p->'requirements_en'->>'recommended')
FROM games
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(g->'platforms') = 'array' THEN g->'platforms' ELSE '[]'::jsonb END
) AS p
ON CONFLICT (game_id, platform_id) DO UPDATE SET
  released_at      = EXCLUDED.released_at,
  requirements_min = EXCLUDED.requirements_min,
  requirements_rec = EXCLUDED.requirements_rec;
"""

SQL_GAME_GENRES = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
)
INSERT INTO public.game_genres (game_id, genre_id)
SELECT DISTINCT
  (g->>'id')::int,
  (x->>'id')::int
FROM games
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(g->'genres') = 'array' THEN g->'genres' ELSE '[]'::jsonb END
) AS x
ON CONFLICT DO NOTHING;
"""

SQL_GAME_STORES = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
)
INSERT INTO public.game_stores (game_id, store_id)
SELECT DISTINCT
  (g->>'id')::int,
  (s->'store'->>'id')::int
FROM games
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(g->'stores') = 'array' THEN g->'stores' ELSE '[]'::jsonb END
) AS s
ON CONFLICT DO NOTHING;
"""

SQL_GAME_TAGS = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
)
INSERT INTO public.game_tags (game_id, tag_id)
SELECT DISTINCT
  (g->>'id')::int,
  (t->>'id')::int
FROM games
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(g->'tags') = 'array' THEN g->'tags' ELSE '[]'::jsonb END
) AS t
ON CONFLICT DO NOTHING;
"""

# Tablas Hijas 1:N
SQL_GAME_RATINGS = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
),
rts AS (
  SELECT
    (g->>'id')::int AS game_id,
    (r->>'id')      AS rid_txt,
    r->>'title'     AS title,
    r->>'count'     AS count_txt,
    r->>'percent'   AS percent_txt
  FROM games
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(g->'ratings') = 'array' THEN g->'ratings' ELSE '[]'::jsonb END
  ) AS r
)
INSERT INTO public.game_ratings_breakdown (game_id, rating_id, title, count, percent)
SELECT
  game_id,
  rid_txt::int,
  title,
  NULLIF(count_txt,'')::int,
  NULLIF(percent_txt,'')::numeric
FROM rts
WHERE rid_txt ~ '^[0-9]+$'
ON CONFLICT (game_id, rating_id) DO UPDATE
SET title = EXCLUDED.title,
    count = EXCLUDED.count,
    percent = EXCLUDED.percent;
"""

SQL_GAME_SCREENSHOTS = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
)
INSERT INTO public.game_screenshots
  (game_id, screenshot_id, image)
SELECT
  (g->>'id')::int,
  (s->>'id')::int,
  s->>'image'
FROM games
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(g->'short_screenshots') = 'array'
       THEN g->'short_screenshots' ELSE '[]'::jsonb END
) AS s
ON CONFLICT (game_id, screenshot_id) DO UPDATE SET
  image = EXCLUDED.image;
"""

SQL_GAME_ADDED_STATUS = """
WITH payload AS (SELECT %s::jsonb AS j),
games AS (
  SELECT jsonb_array_elements(
           CASE WHEN jsonb_typeof(j->'results') = 'array'
                THEN j->'results' ELSE '[]'::jsonb END
         ) AS g
  FROM payload
),
valid AS (
  SELECT g
  FROM games
  WHERE g ? 'added_by_status'
    AND jsonb_typeof(g->'added_by_status') = 'object'
),
kv AS (
  SELECT
    (g->>'id')::int AS game_id,
    each.key        AS status_key,
    CASE WHEN jsonb_typeof(each.value) = 'number' THEN (each.value)::int ELSE NULL END AS count
  FROM valid
  CROSS JOIN LATERAL jsonb_each(
    CASE WHEN jsonb_typeof(g->'added_by_status') = 'object'
         THEN g->'added_by_status' ELSE '{}'::jsonb END
  ) AS each(key, value)
)
INSERT INTO public.game_added_status (game_id, status_key, count)
SELECT game_id, status_key, count
FROM kv
ON CONFLICT (game_id, status_key) DO UPDATE
SET count = EXCLUDED.count;
"""

def process_json_text(conn, json_text: str):
    with conn.cursor() as cur:
        cur.execute("BEGIN;")
        try:
            for name, sql in [
                ("DIM_ESRB", SQL_DIM_ESRB),
                ("DIM_PLATFORMS", SQL_DIM_PLATFORMS),
                ("DIM_GENRES", SQL_DIM_GENRES),
                ("DIM_STORES", SQL_DIM_STORES),
                ("DIM_TAGS", SQL_DIM_TAGS),
                ("GAMES", SQL_GAMES),
                ("GAME_PLATFORMS", SQL_GAME_PLATFORMS),
                ("GAME_GENRES", SQL_GAME_GENRES),
                ("GAME_STORES", SQL_GAME_STORES),
                ("GAME_TAGS", SQL_GAME_TAGS),
                ("GAME_RATINGS", SQL_GAME_RATINGS),
                ("GAME_SCREENSHOTS", SQL_GAME_SCREENSHOTS),
                ("GAME_ADDED_STATUS", SQL_GAME_ADDED_STATUS),
            ]:
                try:
                    cur.execute(sql, (json_text,))
                except Exception as e:
                    raise RuntimeError(f"Fallo en la consulta SQL {name}") from e
            cur.execute("COMMIT;")
        except Exception:
            cur.execute("ROLLBACK;")
            raise

def lambda_handler(event, context):
    files = []

    # Pilla los ficheros de S3 si salta el trigger
    if isinstance(event, dict) and "Records" in event:
        for rec in event["Records"]:
            if rec.get("eventSource") == "aws:s3":
                b = rec["s3"]["bucket"]["name"]
                k = rec["s3"]["object"]["key"]

                if b == bucket_name and k.startswith(f"{s3_prefix}/") and k.endswith(".json"):
                    files.append(k)

    if not files:
        files = get_recent_files()
    if not files:
        return {"status": "no_files"}

    conn = get_connection()
    processed = []
    try:
        for file in files:
            json_text = load_json_from_s3(file)
            process_json_text(conn, json_text)
            processed.append(file)
    finally:
        conn.close()

    return {"status": "ok", "files_processed": processed}


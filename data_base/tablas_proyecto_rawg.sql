-- TABLA PRINCIPAL: GAMES

CREATE TABLE IF NOT EXISTS public.games (
  game_id             INT PRIMARY KEY,
  slug                TEXT,
  name                TEXT NOT NULL,
  released            DATE,
  tba                 BOOLEAN,
  rating              NUMERIC(4,2),
  rating_top          INT,
  ratings_count       INT,
  reviews_text_count  INT,
  metacritic          INT,
  playtime            INT,
  suggestions_count   INT,
  updated             TIMESTAMP,
  background_image    TEXT,
  added               INT,
  reviews_count       INT,
  dominant_color      TEXT,
  saturated_color     TEXT,
  esrb_rating_id      INT REFERENCES public.esrb_ratings(esrb_rating_id),
  website             TEXT
);


 -- TABLAS SECUNDARIAS(dimensiones, listas maestras: información que se repite en varios juegos, tiene identidad propia estable (un id y un nombre), y la vas a filtrar/contar/join a menudo

CREATE TABLE IF NOT EXISTS public.esrb_ratings (
  esrb_rating_id   INT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT
);

CREATE TABLE IF NOT EXISTS public.platforms (
  platform_id   INT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT
);

CREATE TABLE IF NOT EXISTS public.genres (
  genre_id   INT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT
);

CREATE TABLE IF NOT EXISTS public.stores (
  store_id     INT PRIMARY KEY,
  name   TEXT NOT NULL,
  slug   TEXT,
  domain TEXT
);

CREATE TABLE IF NOT EXISTS public.tags (
  tag_id   INT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT
);


-- TABLAS PUENTE N:M: Estas unen games con platforms, genres, stores y tags

CREATE TABLE IF NOT EXISTS public.game_platforms (
  game_id          INT REFERENCES public.games(game_id),
  platform_id      INT REFERENCES public.platforms(platform_id),
  released_at      DATE,
  requirements_min TEXT,
  requirements_rec TEXT,
  PRIMARY KEY (game_id, platform_id)
);

CREATE TABLE IF NOT EXISTS public.game_genres (
  game_id   INT REFERENCES public.games(game_id),
  genre_id  INT REFERENCES public.genres(genre_id),
  PRIMARY KEY (game_id, genre_id)
);

CREATE TABLE IF NOT EXISTS public.game_stores (
  game_id  INT REFERENCES public.games(game_id),
  store_id INT REFERENCES public.stores(store_id),
  PRIMARY KEY (game_id, store_id)
);

CREATE TABLE IF NOT EXISTS public.game_tags (
  game_id INT REFERENCES public.games(game_id),
  tag_id  INT REFERENCES public.tags(tag_id),
  PRIMARY KEY (game_id, tag_id)
);



--- TABLAS HIJAS 1:N


-- 1) ratings[]  (desglose de valoraciones)
CREATE TABLE IF NOT EXISTS public.game_ratings_breakdown (
  game_id    INT REFERENCES public.games(game_id),
  rating_id  INT,              -- ratings[].id (RAWG)
  title      TEXT,             -- ratings[].title  (e.g., exceptional, recommended)
  count      INT,              -- ratings[].count
  percent    NUMERIC(6,3),     -- ratings[].percent
  PRIMARY KEY (game_id, rating_id)
);

-- 2) short_screenshots[]  (imágenes)
CREATE TABLE IF NOT EXISTS public.game_screenshots (
  game_id       INT REFERENCES public.games(game_id),
  screenshot_id INT,           -- short_screenshots[].id (RAWG)
  image         TEXT,          -- short_screenshots[].image (URL)
  PRIMARY KEY (game_id, screenshot_id)
);

-- 3) added_by_status {}  (mapa clave→conteo)
CREATE TABLE IF NOT EXISTS public.game_added_status (
  game_id    INT REFERENCES public.games(game_id),
  status_key TEXT,             -- owned, beaten, toplay, dropped, etc.
  count      INT,
  PRIMARY KEY (game_id, status_key)
);




--USUARIOS extra de prueba y si no tiran con el Admin

-- Crear usuario para Soraya
CREATE USER soraya WITH PASSWORD 'Soraya123!';
CREATE USER lorenzo WITH PASSWORD 'Lorenzo123';

GRANT CONNECT ON DATABASE rawg_db TO soraya;
GRANT CONNECT ON DATABASE rawg_db TO lorenzo;

GRANT USAGE ON SCHEMA public TO soraya;
GRANT USAGE ON SCHEMA public TO lorenzo;


GRANT SELECT ON ALL TABLES IN SCHEMA public TO soraya;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO lorenzo;

-- Permisos
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO soraya;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO lorenzo;

-- privilegios de lectura para futuros objetos
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO soraya;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO lorenzo;

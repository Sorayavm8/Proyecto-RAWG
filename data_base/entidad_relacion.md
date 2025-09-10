```mermaid

erDiagram

  %% Relaciones 1:N
  ESRB_RATINGS ||--o{ GAMES : "1:N (opcional)"
  GAMES ||--o{ GAME_RATINGS_BREAKDOWN : "1:N"
  GAMES ||--o{ GAME_SCREENSHOTS : "1:N"
  GAMES ||--o{ GAME_ADDED_STATUS : "1:N"

  %% Puentes modelados como entidades (cada lado 1:N)
  GAMES ||--o{ GAME_PLATFORMS : "1:N"
  PLATFORMS ||--o{ GAME_PLATFORMS : "1:N"

  GAMES ||--o{ GAME_GENRES : "1:N"
  GENRES ||--o{ GAME_GENRES : "1:N"

  GAMES ||--o{ GAME_STORES : "1:N"
  STORES ||--o{ GAME_STORES : "1:N"

  GAMES ||--o{ GAME_TAGS : "1:N"
  TAGS ||--o{ GAME_TAGS : "1:N"

  ESRB_RATINGS {
    INT esrb_rating_id PK
    STRING name
    STRING slug
  }

  GAMES {
    INT game_id PK
    STRING slug
    STRING name
    DATE released
    BOOLEAN tba
    NUMERIC rating
    INT rating_top
    INT ratings_count
    INT reviews_text_count
    INT metacritic
    INT playtime
    INT suggestions_count
    TIMESTAMP updated
    STRING background_image
    INT added
    INT reviews_count
    STRING dominant_color
    STRING saturated_color
    INT esrb_rating_id FK
    STRING website
  }

  PLATFORMS {
    INT platform_id PK
    STRING name
    STRING slug
  }

  GENRES {
    INT genre_id PK
    STRING name
    STRING slug
  }

  STORES {
    INT store_id PK
    STRING name
    STRING slug
    STRING domain
  }

  TAGS {
    INT tag_id PK
    STRING name
    STRING slug
  }

  GAME_PLATFORMS {
    INT game_id PK, FK
    INT platform_id PK, FK
    DATE released_at
    STRING requirements_min
    STRING requirements_rec
  }

  GAME_GENRES {
    INT game_id PK, FK
    INT genre_id PK, FK
  }

  GAME_STORES {
    INT game_id PK, FK
    INT store_id PK, FK
  }

  GAME_TAGS {
    INT game_id PK, FK
    INT tag_id PK, FK
  }

  GAME_RATINGS_BREAKDOWN {
    INT game_id PK, FK
    INT rating_id PK
    STRING title
    INT count
    NUMERIC percent
  }

  GAME_SCREENSHOTS {
    INT game_id PK, FK
    INT screenshot_id PK
    STRING image
  }

  GAME_ADDED_STATUS {
    INT game_id PK, FK
    STRING status_key PK
    INT count
  }

  ```
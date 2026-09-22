"""Carga el parquet de observaciones de DE-Zrk en la tabla `flux_observations`.

Script SEPARADO del ETL (paso 7). Es autonomo: no importa el paquete `app` del
la app; refleja el esquema ya migrado directamente desde PostgreSQL, de modo
que siempre concuerda con la ultima migracion de Alembic.

Uso (desde la raiz del repo, con el venv de ml activo):

    python -m ml.etl.load_to_db
    python -m ml.etl.load_to_db --database-url postgresql+psycopg2://user:pass@localhost:5432/peattwin
    python -m ml.etl.load_to_db --dry-run

Requisitos: la BD `peattwin` migrada a head (`alembic upgrade head` en la raiz)
y el parquet generado por `python -m ml.etl.run_etl`.

MAPEOS (documentados para el informe):
  flux_observations.nee_co2            <- NEE_F_ANNOPTLM        (gC m-2 d-1)
  flux_observations.fch4               <- FCH4_F_ANNOPTLM       (nmol CH4 m-2 s-1)
  flux_observations.water_table_depth  <- WTD * 100             (m -> cm)
  flux_observations.soil_temp          <- TS_1                  (C, 0.05 m)
  flux_observations.air_temp           <- TA_F                  (C)
  flux_observations.precipitation      <- P_F                   (mm d-1)
  flux_observations.soil_water_content <- NULL                  (no disponible en DE-Zrk)
  flux_observations.carbon_balance_class <- 'SINK' si NEE<0 else 'SOURCE'
  flux_observations.quality_flag       <- FCH4_F_ANNOPTLM_QC    (entero)
  flux_observations.timestamp          <- dia natural anclado a 12:00 UTC

Idempotente: upsert por la restriccion unica (site_id, timestamp).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ml.etl import config as C
from ml import _logging

BATCH = 500


def _database_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    # Lee el .env de la raiz (KEY=VALUE simple).
    if not C.ENV_FILE.exists():
        sys.exit(
            f"No se encuentra {C.ENV_FILE}. Pasa --database-url explicitamente."
        )
    env: dict[str, str] = {}
    for line in C.ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    try:
        return (
            f"postgresql+psycopg2://{env['POSTGRES_USER']}:{env['POSTGRES_PASSWORD']}"
            f"@{env['POSTGRES_HOST']}:{env['POSTGRES_PORT']}/{env['POSTGRES_DB']}"
        )
    except KeyError as exc:  # pragma: no cover
        sys.exit(f"Falta la clave {exc} en {C.ENV_FILE}")


def _build_rows(df: pd.DataFrame, site_id: int) -> list[dict]:
    # El registro diario representa un dia natural. Se ancla a las 12:00 UTC:
    # asi `timestamp::date` devuelve el dia correcto en CUALQUIER zona horaria de
    # sesion (mediodia UTC +/- 14 h no cruza de dia). El desfase de hora local
    # del sitio (UTC+1) queda solo en los metadatos; no se aplica a un agregado
    # diario.
    ts = (pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=12)).dt.tz_localize("UTC")
    nee = df[C.TARGET_NEE]
    rows: list[dict] = []
    for i in range(len(df)):
        rows.append(
            {
                "site_id": site_id,
                "timestamp": ts.iloc[i].to_pydatetime(),
                "nee_co2": _f(nee.iloc[i]),
                "fch4": _f(df[C.TARGET_FCH4].iloc[i]),
                "water_table_depth": _f(df["WTD"].iloc[i], scale=C.WTD_METERS_TO_CM),
                "soil_temp": _f(df["TS_1"].iloc[i]),
                "soil_water_content": None,  # no disponible en este dataset
                "air_temp": _f(df["TA_F"].iloc[i]),
                "precipitation": _f(df["P_F"].iloc[i]),
                "carbon_balance_class": (
                    None
                    if pd.isna(nee.iloc[i])
                    else ("SINK" if nee.iloc[i] < 0 else "SOURCE")
                ),
                "quality_flag": _i(df.get("FCH4_F_ANNOPTLM_QC", pd.Series()).get(i)),
            }
        )
    return rows


def _f(v, scale: float = 1.0):
    return None if pd.isna(v) else float(v) * scale


def _i(v):
    return None if v is None or pd.isna(v) else int(round(float(v)))


def main() -> None:
    _logging.setup("carga_bd")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database-url", default=None, help="URL SQLAlchemy de PostgreSQL")
    ap.add_argument("--parquet", default=str(C.OUT_OBSERVATIONS))
    ap.add_argument("--dry-run", action="store_true", help="No escribe en la BD")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, engine="pyarrow")
    print(f"parquet: {args.parquet}  ({len(df)} filas)")

    engine = create_engine(_database_url(args.database_url), future=True)
    md = MetaData()
    sites = Table("sites", md, autoload_with=engine)
    flux = Table("flux_observations", md, autoload_with=engine)

    with engine.begin() as conn:
        # get-or-create del sitio DE-Zrk
        site_id = conn.execute(
            select(sites.c.id).where(sites.c.fluxnet_id == C.SITE["fluxnet_id"])
        ).scalar_one_or_none()

        if site_id is None:
            if args.dry_run:
                print(f"[dry-run] crearia el sitio {C.SITE['fluxnet_id']}")
                site_id = -1
            else:
                site_id = conn.execute(
                    pg_insert(sites)
                    .values(
                        name=C.SITE["name"],
                        fluxnet_id=C.SITE["fluxnet_id"],
                        peatland_type=C.SITE["peatland_type"],
                        latitude=C.SITE["latitude"],
                        longitude=C.SITE["longitude"],
                        description="FLUXNET-CH4 DE-Zrk; fen rehumedecido 2004-2005",
                    )
                    .returning(sites.c.id)
                ).scalar_one()
                print(f"sitio creado: id={site_id}")
        else:
            print(f"sitio existente: id={site_id}")

        rows = _build_rows(df, site_id)

        if args.dry_run:
            print(f"[dry-run] upsert de {len(rows)} filas omitido. Ejemplo:")
            print(rows[0])
            return

        upserted = 0
        for start in range(0, len(rows), BATCH):
            chunk = rows[start : start + BATCH]
            stmt = pg_insert(flux).values(chunk)
            update_cols = {
                c: stmt.excluded[c]
                for c in (
                    "nee_co2",
                    "fch4",
                    "water_table_depth",
                    "soil_temp",
                    "soil_water_content",
                    "air_temp",
                    "precipitation",
                    "carbon_balance_class",
                    "quality_flag",
                )
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_flux_obs_site_timestamp", set_=update_cols
            )
            conn.execute(stmt)
            upserted += len(chunk)

        total = conn.execute(
            select(func.count()).select_from(flux).where(flux.c.site_id == site_id)
        ).scalar_one()

    print(f"OK. upsert de {upserted} filas en flux_observations (site_id={site_id}).")
    print(f"filas totales del sitio en la tabla: {total}")


if __name__ == "__main__":
    main()
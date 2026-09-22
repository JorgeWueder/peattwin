# PeatTwin - Datos

- `raw/`       : datos originales sin modificar (telemetria de campo, sensores de
                 nivel freatico, flujos de CO2/CH4, series meteorologicas...).
                 Nunca se editan a mano.
- `processed/` : datasets limpios y listos para entrenamiento/inferencia,
                 generados por scripts del modulo `/ml`.

El contenido de ambas carpetas esta excluido del control de versiones
(solo se versiona `.gitkeep`).

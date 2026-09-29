"""Capa visual de PeatTwin: tokens de diseno y CSS de los componentes.

Un unico `inject()` al arrancar, desde `streamlit_app.py`. Este modulo NO
renderiza contenido ni toca estado: solo aspecto. Quitarlo deja la aplicacion
funcionando igual, con el tema por defecto de Streamlit.

DIRECCION
---------
Instrumento de medicion, no panel generico. La referencia son paneles de datos
reales (Linear, Vercel, Stripe): neutros zinc frios, un solo acento, escala
tipografica corta, numeros en monoespaciada tabular y separacion por hairlines
en lugar de cajas. En un panel denso las tarjetas son ruido: agrupan por marco
lo que ya agrupa la rejilla.

MOVIMIENTO
----------
Reglas de Emil Kowalski, aplicadas literalmente:
  - Solo se animan `transform` y `opacity` (mas color en hover, que es pintado).
    Nunca `width`, `height`, `margin` ni `padding`.
  - Nunca `transition: all`: cada transicion nombra su propiedad.
  - Nunca `ease-in`, que arranca lento justo cuando el usuario mira. Las
    entradas van con la curva fuerte `--pt-ease-out`.
  - Todo por debajo de 300 ms; la pulsacion, en 140.
  - El hover va detras de `@media (hover: hover)`: en tactil un toque dispara
    hover y deja el estado pegado.

Decision deliberada: NO hay animacion de entrada en cascada (stagger) de los
bloques de la pagina. Streamlit vuelve a montar el arbol en CADA rerun, asi que
una cascada de entrada se repetiria con cada tick del slider del simulador y con
cada cambio de fecha. Por la tabla de frecuencia de Emil eso cae en «decenas de
veces al dia, reducir o eliminar»: aqui se elimina.

ALCANCE
-------
Los selectores usan `data-testid`, que es la superficie estable de Streamlit;
los `st-emotion-cache-*` cambian entre versiones y no se referencian. Si una
version futura renombra un testid, se pierde ese detalle visual y el resto
sigue en pie: ninguna regla de aqui es necesaria para que la app funcione.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
/* ==================================================== tokens */
:root {
  /* Curvas de Emil: las de CSS (`ease-out`, `ease`) son demasiado flojas y
     dejan la interfaz sin intencion. */
  --pt-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --pt-ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);

  --pt-dur-press: 140ms;   /* pulsacion: rango 100-160 */
  --pt-dur-hover: 160ms;
  --pt-dur-panel: 200ms;   /* desplegables: rango 150-250 */

  --pt-accent: #059669;
  --pt-accent-soft: rgba(5, 150, 105, 0.10);
  --pt-accent-line: rgba(5, 150, 105, 0.32);

  --pt-ink: #18181b;
  --pt-muted: #71717a;
  --pt-faint: #a1a1aa;
  --pt-border: #e4e4e7;
  --pt-border-strong: #d4d4d8;
  --pt-surface: #ffffff;
  --pt-surface-sunk: #f4f4f5;

  --pt-mono: "Geist Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;

  /* Micro-etiqueta: mayusculas, 11px, tracking abierto. Es el unico recurso
     de etiqueta pequena del sistema y se usa solo en la tira de metricas. */
  --pt-label-size: 0.6875rem;
  --pt-label-track: 0.07em;
}

/* Mismo juego de tokens en oscuro. Sin negro puro: mata la profundidad. */
@media (prefers-color-scheme: dark) {
  :root {
    --pt-ink: #fafafa;
    --pt-muted: #a1a1aa;
    --pt-faint: #71717a;
    --pt-border: #27272a;
    --pt-border-strong: #3f3f46;
    --pt-surface: #111113;
    --pt-surface-sunk: #18181b;
    --pt-accent: #10b981;
    --pt-accent-soft: rgba(16, 185, 129, 0.12);
    --pt-accent-line: rgba(16, 185, 129, 0.38);
  }
}

/* ==================================================== reduced motion
   No es «cero animacion»: se conservan opacidad y color, que ayudan a
   entender el cambio de estado, y se retira el desplazamiento. */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-property: opacity, color, background-color, border-color !important;
    transition-duration: 120ms !important;
  }
  *:active { transform: none !important; }
}

/* ==================================================== ritmo de pagina
   El defecto de Streamlit centra el contenido en una columna estrecha con
   mucho aire arriba. Un panel de datos necesita lo contrario: ancho util y
   una cabecera que no flote. */
[data-testid="stMainBlockContainer"] {
  padding-top: 2.75rem;
  padding-bottom: 5rem;
  max-width: 1480px;
}

/* ==================================================== tipografia */
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2 {
  letter-spacing: -0.021em;   /* el texto grande necesita tracking negativo */
  line-height: 1.15;
}
[data-testid="stHeading"] h3,
[data-testid="stHeading"] h4 {
  letter-spacing: -0.012em;
}

/* Ritmo vertical. El defecto de Streamlit deja el mismo hueco entre todo, asi
   que una seccion nueva no se distingue de un parrafo mas. Se abre el espacio
   ANTES del encabezado y se cierra el que va DESPUES: el titulo queda pegado
   a lo que titula, que es lo que agrupa sin necesidad de una caja.
   `:not(:first-child)` evita meter el hueco en el primer bloque de la vista. */
[data-testid="stMain"] [data-testid="stElementContainer"]:has(> [data-testid="stHeading"] h2):not(:first-child) {
  margin-top: 2.5rem;
}
[data-testid="stMain"] [data-testid="stElementContainer"]:has(> [data-testid="stHeading"] h3):not(:first-child) {
  margin-top: 1.75rem;
}
[data-testid="stMain"] [data-testid="stHeading"]:has(h2),
[data-testid="stMain"] [data-testid="stHeading"]:has(h3) {
  margin-bottom: 0.125rem;
}

/* El h1 de cada vista lleva un filete de acento a la izquierda: ancla la
   cabecera al borde del contenido y da identidad sin recurrir a un eyebrow
   (la etiqueta en mayusculas sobre cada titulo es justo el tic que satura
   estas interfaces). */
/* `h1` no es hijo directo de `stHeading`: cuelga de
   stHeading > stMarkdownContainer > stHeadingWithActionElements > h1. */
[data-testid="stMain"] [data-testid="stHeading"]:has(h1) {
  border-left: 2px solid var(--pt-accent);
  padding-left: 0.875rem;
  margin-left: -0.875rem;
}

[data-testid="stCaptionContainer"] {
  color: var(--pt-muted);
  line-height: 1.55;
  max-width: 78ch;   /* medida de lectura: el ancho total cansa */
}

/* Codigo y rutas en mono, un punto mas pequeno y sobre fondo hundido. */
[data-testid="stMarkdownContainer"] code {
  font-family: var(--pt-mono);
  font-size: 0.86em;
  background: var(--pt-surface-sunk);
  border: 1px solid var(--pt-border);
  border-radius: 4px;
  padding: 0.08em 0.34em;
}

/* ==================================================== tira de metricas
   El movimiento principal del rediseno. Antes: seis numeros sueltos flotando
   con etiqueta gris, sin jerarquia ni agrupacion, y el sexto recortado
   («Clase ac...», «8...»). Ahora: lectura de instrumento. Etiqueta en
   mayusculas pequenas, valor en mono tabular (las cifras no bailan al
   cambiar) y hairline entre columnas.

   Sin tarjetas, a proposito: en un panel denso la caja generica esta vetada y
   los datos deben respirar en rejilla limpia. El selector exige que la
   metrica sea hija directa de la columna, para no tocar las columnas de
   maquetacion del simulador, que solo la contienen anidada. */
[data-testid="stHorizontalBlock"]:has(
  > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
  > [data-testid="stElementContainer"] > [data-testid="stMetric"]
) {
  border-top: 1px solid var(--pt-border);
  border-bottom: 1px solid var(--pt-border);
  padding: 0.875rem 0;
  margin: 0.5rem 0 1.75rem;
  gap: 0 !important;
}

[data-testid="stHorizontalBlock"]:has(
  > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
  > [data-testid="stElementContainer"] > [data-testid="stMetric"]
) > [data-testid="stColumn"] {
  padding: 0 1.125rem;
}

/* Hairline solo entre columnas, nunca antes de la primera. */
[data-testid="stHorizontalBlock"]:has(
  > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
  > [data-testid="stElementContainer"] > [data-testid="stMetric"]
) > [data-testid="stColumn"] + [data-testid="stColumn"] {
  border-left: 1px solid var(--pt-border);
}

/* SIN `text-transform: uppercase`, y es deliberado. Estas etiquetas llevan
   unidades: `gC m-2 d-1` en mayusculas queda `GC M-2 D-1`, y GC no es gC ni M
   es m. Convertir a mayusculas corrompe el dato, asi que el registro de
   micro-etiqueta se consigue solo con tamano, peso, color y tracking. */
[data-testid="stMetricLabel"] {
  letter-spacing: var(--pt-label-track);
  font-size: var(--pt-label-size);
  font-weight: 500;
  color: var(--pt-muted);
  /* Reserva de dos lineas: con etiquetas de distinta longitud en la misma
     tira, los valores tienen que caer a la misma altura. */
  min-height: 2.15em;
  display: flex;
  align-items: flex-start;
}
/* La etiqueta puede ocupar dos lineas antes que recortarse: perder «Clase
   acierto» en «Clase ac...» es perder el dato. */
[data-testid="stMetricLabel"] * {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}

[data-testid="stMetricValue"] {
  font-family: var(--pt-mono);
  font-feature-settings: "tnum" 1, "zero" 1;   /* tabular + cero con barra */
  font-size: 1.5rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  line-height: 1.2;
  color: var(--pt-ink);
}

/* Reflujo de la tira de metricas.
   Streamlit no apila estas columnas hasta muy estrecho, asi que con seis
   metricas por debajo de ~1200px cada columna baja a 80px y Streamlit recorta
   el propio valor con puntos suspensivos: «1 . …», «43…», «83…». Un numero a
   medias es peor que el diseno de partida, asi que a partir de ese ancho la
   tira pasa a varias filas en vez de comprimirse. Se retiran los hairlines
   verticales, que al envolver aparecerian al principio de cada fila. */
@media (max-width: 1200px) {
  [data-testid="stHorizontalBlock"]:has(
    > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
    > [data-testid="stElementContainer"] > [data-testid="stMetric"]
  ) {
    flex-wrap: wrap;
    row-gap: 1.25rem !important;
  }
  [data-testid="stHorizontalBlock"]:has(
    > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
    > [data-testid="stElementContainer"] > [data-testid="stMetric"]
  ) > [data-testid="stColumn"] {
    flex: 1 1 9rem;
    min-width: 9rem;
    padding: 0 1rem 0 0;
  }
  [data-testid="stHorizontalBlock"]:has(
    > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
    > [data-testid="stElementContainer"] > [data-testid="stMetric"]
  ) > [data-testid="stColumn"] + [data-testid="stColumn"] {
    border-left: none;
  }
}

/* Ultima linea de defensa: el valor de una metrica nunca se recorta. Antes que
   mostrar «83…» donde pone «83 %», se deja desbordar.

   Pero `overflow: visible` a secas tiene un precio que costo un fallo real: un
   valor de TEXTO largo —«Stacking Ensemble (tuned)», 298px en una columna de
   173px— no se recortaba pero se pintaba ENCIMA de la metrica siguiente, y el
   nombre del modelo y su score salian superpuestos. Que no se corte no puede
   significar que invada al vecino.

   La salida es envolver en vez de desbordar: `break-word` solo parte cuando la
   palabra no cabe por si sola, asi que un numero —que nunca tiene espacios y
   siempre cabe— se sigue mostrando entero y en una linea, y una cadena con
   espacios baja de renglon dentro de SU columna. La reserva de dos lineas de
   la etiqueta mantiene los valores alineados aunque uno ocupe dos. */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] {
  overflow: visible !important;
  text-overflow: clip !important;
  white-space: normal !important;
  overflow-wrap: break-word;
}

/* ==================================================== bloques de interpretacion
   `ui.interpret._bloque`: la regla del proyecto es que ningun numero se muestra
   solo, y que la anotacion va en DOS partes separadas —explicacion (que es) e
   interpretacion (que significa)—. Eso merece forma propia, no otra tarjeta
   igual a las demas: filete de acento a la izquierda y fondo apenas tenido,
   para que se lea como anotacion al margen del dato que acompana.

   El ancla es la clase `st-key-ptnotaN` que Streamlit genera a partir de la
   `key` del contenedor. Antes se enganchaba a «contenedor con borde cuyo unico
   hijo es un caption», y eso dejo de ser cierto al pasar de uno a dos captions:
   un selector que describe el contenido se rompe en cuanto el contenido cambia,
   mientras que una key puesta a proposito no.

   CUIDADO: en esta hoja no puede aparecer NUNCA el caracter «menor que», ni
   siquiera dentro de un comentario. `inject()` envuelve la hoja en una etiqueta
   de estilo y la entrega por `st.html`, cuyo saneador parsea ese contenido como
   HTML: un solo signo de menor suelto le hace creer que empieza una etiqueta y
   descarta la hoja ENTERA, en silencio, sin error ni en el servidor ni en la
   consola del navegador. Costo un rato encontrarlo: este mismo comentario
   nombraba la clase como «ptnota» seguida de la letra n entre signos de mayor y
   menor, y por ese unico caracter la aplicacion se quedo sin sus 23 KB de
   estilos conservando solo los del login, que se inyectan aparte. La funcion
   `_valida_css()` de este modulo lo comprueba ahora al importar. */
[data-testid="stMain"] [class*="st-key-ptnota"] {
  border: 1px solid var(--pt-border) !important;
  border-left: 2px solid var(--pt-accent-line) !important;
  background: var(--pt-accent-soft);
  border-radius: 0 6px 6px 0 !important;
  padding: 0.75rem 1rem !important;
  margin-top: 0.375rem;
}

/* El texto llena la caja. Antes llevaba `max-width: 88ch` heredando ademas el
   `78ch` de la regla general de captions, y el resultado era que la caja media
   864px mientras el texto se cortaba a 509-574px: 300px de vacio a la derecha
   de cada interpretacion, y con anchos distintos entre cajas porque `ch`
   depende de la fuente y las que llevan `code` inline resuelven otra.

   Una medida de lectura es buena tipografia, pero aqui competia con el borde:
   la caja anunciaba un ancho que el texto no usaba, y eso se lee como error de
   maquetacion, no como respiro. Entre estrechar la caja o ensanchar el texto,
   gana el texto: estas anotaciones acompanan a una tabla o un grafico y deben
   alinearse con el ancho de lo que explican. */
[data-testid="stMain"] [class*="st-key-ptnota"] [data-testid="stCaptionContainer"] {
  color: var(--pt-ink);
  opacity: 0.82;
  max-width: none;
}

/* Separacion entre las dos partes. Un filete tenue en vez de mas espacio en
   blanco: el bloque ya vive dentro de una caja, y solo con aire las dos partes
   se leerian como un parrafo partido en vez de como dos respuestas distintas.

   Los `stElementContainer` cuelgan DIRECTAMENTE del contenedor con la key: no
   hay `stVerticalBlock` intermedio. Se comprobo en el DOM porque la primera
   version daba por supuesto ese nivel de mas y no casaba con nada. */
[data-testid="stMain"] [class*="st-key-ptnota"]
  > [data-testid="stElementContainer"]:not(:first-child) {
  border-top: 1px solid var(--pt-border);
  padding-top: 0.55rem;
  margin-top: 0.55rem;
}

/* El rotulo de cada parte («Explicacion.» / «Interpretacion.») va en la
   micro-etiqueta del sistema, el mismo registro que las etiquetas de las
   metricas: es un rotulo de estructura, no enfasis dentro de la frase. */
/* Las clases de emotion que genera Streamlit ganan por especificidad, asi que
   este rotulo necesita `!important` para imponerse: sin el, el selector casaba
   (12 nodos) pero el `display: block` y el color se perdian en la cascada y las
   dos partes salian como un parrafo corrido con una palabra en negrita. */
[data-testid="stMain"] [class*="st-key-ptnota"] [data-testid="stCaptionContainer"] strong {
  display: block !important;
  font-size: var(--pt-label-size) !important;
  letter-spacing: var(--pt-label-track) !important;
  font-weight: 600 !important;
  color: var(--pt-accent) !important;
  opacity: 1 !important;
  margin-bottom: 0.2rem;
}

/* ==================================================== botones
   Pulsacion con `scale(0.97)`: es lo que hace que la interfaz parezca estar
   escuchando. Hover solo en puntero fino. */
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
  transition:
    transform var(--pt-dur-press) var(--pt-ease-out),
    background-color var(--pt-dur-hover) ease,
    border-color var(--pt-dur-hover) ease,
    color var(--pt-dur-hover) ease;
  font-weight: 500;
  letter-spacing: -0.005em;
}
[data-testid="stButton"] button:active,
[data-testid="stDownloadButton"] button:active,
[data-testid="stFormSubmitButton"] button:active {
  transform: scale(0.97);
}
@media (hover: hover) and (pointer: fine) {
  [data-testid="stButton"] button:hover,
  [data-testid="stDownloadButton"] button:hover,
  [data-testid="stFormSubmitButton"] button:hover {
    border-color: var(--pt-accent);
  }
}

/* Foco visible y con el color del sistema. No se debilita ninguno de los
   estados de foco que ya traia Streamlit: solo se unifica su aspecto. */
[data-testid="stMain"] button:focus-visible,
[data-testid="stSidebar"] button:focus-visible,
[data-testid="stMain"] a:focus-visible {
  outline: 2px solid var(--pt-accent);
  outline-offset: 2px;
}

/* ==================================================== campos de entrada */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stDateInput"] div[data-baseweb="input"],
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stTextInput"] div[data-baseweb="input"] {
  transition: border-color var(--pt-dur-hover) ease,
              box-shadow var(--pt-dur-hover) ease;
}
@media (hover: hover) and (pointer: fine) {
  [data-testid="stSelectbox"] div[data-baseweb="select"]:hover > div,
  [data-testid="stDateInput"] div[data-baseweb="input"]:hover,
  [data-testid="stMultiSelect"] div[data-baseweb="select"]:hover > div,
  [data-testid="stTextInput"] div[data-baseweb="input"]:hover {
    border-color: var(--pt-border-strong);
  }
}

[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {
  font-size: 0.8125rem;
  font-weight: 500;
  color: var(--pt-muted);
  letter-spacing: 0;
}

/* Las fechas son datos: en mono, como el resto de las cifras. */
[data-testid="stDateInputField"] input {
  font-family: var(--pt-mono);
  font-feature-settings: "tnum" 1;
  font-size: 0.875rem;
}

/* El valor que sigue al pulgar del slider tambien es una cifra que cambia. */
[data-testid="stSliderThumbValue"],
[data-testid="stSliderTickBar"] {
  font-family: var(--pt-mono);
  font-feature-settings: "tnum" 1;
  font-size: 0.75rem;
}
[data-testid="stSlider"] [role="slider"] {
  transition: transform var(--pt-dur-press) var(--pt-ease-out),
              box-shadow var(--pt-dur-hover) ease;
}
[data-testid="stSlider"] [role="slider"]:active {
  transform: scale(1.12);   /* el pulgar crece al agarrarlo, no se encoge */
}
[data-testid="stSlider"] [role="slider"]:focus-visible {
  outline: 2px solid var(--pt-accent);
  outline-offset: 3px;
}

/* ==================================================== barra lateral */
[data-testid="stSidebarNavLink"] {
  transition: background-color var(--pt-dur-hover) ease,
              color var(--pt-dur-hover) ease,
              transform var(--pt-dur-hover) var(--pt-ease-out);
  border-radius: 6px;
  border-left: 2px solid transparent;
}
@media (hover: hover) and (pointer: fine) {
  [data-testid="stSidebarNavLink"]:hover {
    transform: translateX(2px);   /* transform, no margin: no toca el layout */
  }
}
/* Pagina activa: filete de acento, el mismo recurso que la cabecera h1. */
[data-testid="stSidebarNavLink"][aria-current="page"] {
  border-left-color: var(--pt-accent);
  font-weight: 500;
}
[data-testid="stSidebarNavItems"] {
  gap: 0.0625rem;
}
[data-testid="stSidebarNavLink"] span {
  font-size: 0.8125rem;
  letter-spacing: -0.003em;
}

/* Jerarquia de la barra lateral: navegacion arriba, identidad abajo.
   Streamlit fija el ORDEN de los slots (cabecera, navegacion, contenido de
   usuario) y no se puede cambiar desde Python: `_sidebar()` escribe antes que
   `st.navigation`, y aun asi la navegacion se pinta primero. Lo que si se
   puede es anclar el bloque de identidad al fondo, que es donde se espera
   («quien soy» y «salir» no compiten con el menu).
   Nada se mueve a un menu oculto: la marca, el usuario, su rol y el boton de
   cerrar sesion siguen visibles, solo se agrupan como pie. */
[data-testid="stSidebarContent"] {
  display: flex;
  flex-direction: column;
}
[data-testid="stSidebarNav"] {
  flex: 0 0 auto;
}
[data-testid="stSidebarUserContent"] {
  margin-top: auto;      /* empuja el grupo al fondo */
  flex: 0 0 auto;
  padding-bottom: 1rem;
}

/* Marca: lockup contenido, no titulo de seccion. */
[data-testid="stSidebarUserContent"] h3 {
  font-size: 0.9375rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin-bottom: 0.125rem;
}
[data-testid="stSidebarUserContent"] [data-testid="stCaptionContainer"] {
  font-size: 0.75rem;
}

/* El identificador de cuenta es un dato: en mono, como el resto. */
[data-testid="stSidebarUserContent"] hr ~ div [data-testid="stCaptionContainer"] p {
  font-family: var(--pt-mono);
  font-size: 0.6875rem;
  letter-spacing: -0.01em;
}

[data-testid="stSidebar"] hr {
  margin: 1rem 0 0.75rem;
  border: none;
  border-top: 1px solid var(--pt-border);
}

/* ==================================================== tablas y graficos
   Marco de 1px y radio del sistema. La tabla es el dato en bruto: se le da
   contencion, no decoracion. */
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"] {
  border-radius: 8px;
  overflow: hidden;
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--pt-border);
}

[data-testid="stVegaLiteChart"] {
  border: 1px solid var(--pt-border);
  border-radius: 8px;
  padding: 0.875rem 0.75rem 0.375rem;
  background: var(--pt-surface);
}

/* Las imagenes de figuras del pipeline comparten el mismo marco, para que
   figura PNG y grafico Altair se lean como el mismo objeto. */
[data-testid="stImage"] img {
  border: 1px solid var(--pt-border);
  border-radius: 8px;
}

/* ==================================================== listas de permisos
   Las tarjetas de rol de «Usuarios y roles» imprimen sus permisos con
   `st.code`, que no envuelve: el rol admin ocupa 1171px dentro de una columna
   de 453px, asi que casi dos tercios de sus permisos quedaban tras un scroll
   horizontal. La tarjeta existe para ver el conjunto de un vistazo, asi que
   aqui se permite el salto de linea. Se muestra MAS dato, no menos.

   El selector describe la FORMA de la tarjeta de rol: dentro de una columna,
   un contenedor con exactamente tres elementos (nombre, descripcion y la
   lista de codigos), siendo el tercero el ultimo.
   Asi queda fuera el visor de logs de Motor IA, que es un `st.code` suelto en
   el bloque raiz de la pagina y no vive en ninguna columna: ahi envolver las
   lineas romperia la lectura del log. Un primer intento pedia solo «un bloque
   con un code y un caption», y eso encajaba tambien con el bloque raiz de la
   pagina de logs, que tiene los dos entre sus descendientes. */
[data-testid="stColumn"] [data-testid="stVerticalBlock"]:has(
  > [data-testid="stElementContainer"]:nth-child(3):last-child
) [data-testid="stCode"] :is(pre, code) {
  /* Hace falta el `code` y no solo el `pre`: el `pre` envuelve, pero el
     `code` de dentro mantiene `white-space: pre` y se estira a 1156px.
     `!important` porque la regla propia de Streamlit sobre el `code` gana en
     especificidad incluso con este selector. */
  white-space: pre-wrap !important;
  word-break: break-word;
  line-height: 1.7;
}

/* ==================================================== desplegables */
[data-testid="stExpander"] details {
  border: 1px solid var(--pt-border);
  border-radius: 8px;
  transition: border-color var(--pt-dur-hover) ease;
}
@media (hover: hover) and (pointer: fine) {
  [data-testid="stExpander"] details:hover {
    border-color: var(--pt-border-strong);
  }
}
[data-testid="stExpander"] summary {
  transition: background-color var(--pt-dur-hover) ease;
  font-weight: 500;
  font-size: 0.875rem;
}
[data-testid="stExpander"] summary:focus-visible {
  outline: 2px solid var(--pt-accent);
  outline-offset: -2px;
}

/* ==================================================== control segmentado
   Navegacion interna de Motor IA. Diez botones que se pulsan a menudo: la
   respuesta tiene que ser inmediata, sin desplazamiento. */
[data-testid="stButtonGroup"] button {
  transition: transform var(--pt-dur-press) var(--pt-ease-out),
              background-color var(--pt-dur-hover) ease,
              color var(--pt-dur-hover) ease;
  font-size: 0.8125rem;
  font-weight: 500;
}
[data-testid="stButtonGroup"] button:active {
  transform: scale(0.98);
}

/* ==================================================== avisos
   Solo el radio del sistema. Sin filete lateral de color: Streamlit ya tine el
   fondo segun la severidad, asi que el filete no anade informacion y el borde
   grueso de un lado es uno de los tics mas reconocibles de interfaz generada
   (lo marca el detector de Impeccable como antipatron `side-tab`).
   El filete de acento se reserva para dos usos con significado propio: la
   cabecera de cada vista y los bloques de interpretacion. */
[data-testid="stAlert"],
[data-testid="stAlertContainer"] {
  border-radius: 8px;
}

/* ==================================================== separadores
   `st.divider()` por defecto es una linea al 100% que parte la pagina. Aqui
   separa sin cortar. */
[data-testid="stMain"] hr {
  border: none;
  border-top: 1px solid var(--pt-border);
  margin: 2.25rem 0 1.5rem;
}
"""


# ===================================================================== acceso
# La pantalla de acceso es la unica superficie de la aplicacion que no es un
# panel de datos, y por tanto la unica donde aplican las reglas de composicion
# de una pagina de entrada. Se sirve aparte porque el ancho de 1480px que
# necesita un dashboard deja el formulario estirado de lado a lado.
#
# Columna contenida y alineada a la izquierda, no tarjeta centrada en medio de
# la ventana: el bloque arranca en el tercio superior, que es donde cae la
# mirada, y el texto comparte eje con el campo que hay debajo.
_CSS_LOGIN = """
/* Sin sesion la barra lateral existe pero esta VACIA: la navegacion va oculta
   (`position="hidden"`) y no hay bloque de usuario, asi que solo queda su
   flecha de colapso sobre 300px de banda. Se oculta el contenedor vacio; no
   hay ningun control dentro que se pierda. */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"] {
  display: none !important;
}

[data-testid="stMainBlockContainer"] {
  max-width: 30rem;
  padding-top: 7vh;
  padding-bottom: 4rem;
}

/* En la vista de acceso el titulo SI es el elemento principal de la pagina,
   asi que recupera escala. En las vistas de datos compite con las cifras y se
   queda en 1.75rem.
   Ojo: aqui el titulo sale de `st.markdown("## ...")`, que NO pasa por
   `stHeading`, sino por `stMarkdownContainer`. */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2 {
  font-size: 2rem;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin-bottom: 0.25rem;
  border-left: 2px solid var(--pt-accent);
  padding-left: 0.875rem;
  margin-left: -0.875rem;
}

[data-testid="stMain"] [data-testid="stCaptionContainer"] {
  max-width: 100%;
}

/* El formulario recibe marco propio: en una pagina con un unico bloque, la
   contencion es lo que le da centro de gravedad. */
[data-testid="stForm"] {
  border: 1px solid var(--pt-border);
  border-radius: 10px;
  padding: 1.25rem;
  background: var(--pt-surface);
}

/* Boton de envio a todo el ancho del formulario. Hay que estirar tambien el
   `stElementContainer`, que viene ajustado al contenido y es quien encoge al
   boton a sus 61px. */
[data-testid="stForm"] [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) {
  width: 100% !important;
}
[data-testid="stFormSubmitButton"],
[data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
  width: 100% !important;
}
[data-testid="stFormSubmitButton"] {
  margin-top: 0.375rem;
}

/* Pestanas Entrar / Crear una cuenta: subrayado fino y transicion de color,
   sin desplazamiento. */
[data-testid="stTabs"] button[role="tab"] {
  transition: color var(--pt-dur-hover) ease;
  font-size: 0.875rem;
  font-weight: 500;
}
"""


_ABRE_ETIQUETA = chr(60)   # el signo de «menor que», escrito asi a proposito:
                           # ponerlo literal en este fichero seria inofensivo en
                           # el codigo, pero invita a copiarlo dentro de la hoja.


def _valida_css() -> None:
    """Falla al importar si alguna hoja lleva un caracter que la anularia.

    `st.html` sanea su contenido como HTML. Un signo de «menor que» dentro del
    CSS —aunque este en un comentario— hace que el saneador se coma la hoja
    entera sin avisar: la aplicacion arranca, no hay excepcion ni warning, y
    simplemente sale sin estilos. Un fallo mudo que se descubre mirando pixeles
    es caro, asi que aqui se convierte en un error al arrancar, que es barato.
    """
    for nombre, hoja in (("_CSS", _CSS), ("_CSS_LOGIN", _CSS_LOGIN)):
        if _ABRE_ETIQUETA in hoja:
            pos = hoja.index(_ABRE_ETIQUETA)
            raise ValueError(
                f"{nombre} contiene un signo de «menor que» en la posicion {pos}, "
                f"que anularia la hoja entera al pasar por st.html. "
                f"Contexto: ...{hoja[max(0, pos - 60):pos + 20]!r}..."
            )


_valida_css()


def inject() -> None:
    """Escribe la hoja de estilos base. Idempotente por rerun."""
    st.html(f"{_ABRE_ETIQUETA}style>{_CSS}{_ABRE_ETIQUETA}/style>")


def inject_login() -> None:
    """Estilos adicionales de la pantalla de acceso. Solo CSS, como `inject`."""
    st.html(f"{_ABRE_ETIQUETA}style>{_CSS_LOGIN}{_ABRE_ETIQUETA}/style>")

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np

# ── Parámetros de estandarización ──────────────────────────────────────────
MEANS = {"age": 43.6123996, "glucose": 106.3429493, "bmi": 28.58072289}
STDS  = {"age": 22.62617431, "glucose": 45.36217274, "bmi": 6.75895657}

# ── Coeficientes del modelo EDA Reducido ───────────────────────────────────
COEF = {
    "intercept":                          -4.2466,
    "age":                                 1.6781,
    "avg_glucose_level":                   0.2146,
    "bmi":                                 0.2693,
    "hypertension_1":                      0.5398,
    "work_type_Govt_job":                 -0.2866,
    "work_type_Self_employed":            -0.3081,
    "work_type_children":                  1.7988,
    "smoking_status_formerly_smoked":      0.2765,
    "smoking_status_smokes":               0.1171,
    "smoking_status_Unknown":              0.1736,
    "heart_disease_1":                    -0.0745,
    "age:bmi":                            -0.2104,
    "heart_disease_1:smoking_status_smokes": 1.2317,
}

# ── Orden de variables para el vector x ───────────────────────────────────
VAR_ORDER = [
    "Intercept", "age", "avg_glucose_level", "bmi", "hypertension_1",
    "work_type_Govt_job", "work_type_Self_employed", "work_type_children",
    "smoking_status_formerly_smoked", "smoking_status_smokes",
    "smoking_status_Unknown", "heart_disease_1", "age:bmi",
    "heart_disease_1:smoking_status_smokes"
]

# ── Matriz de covarianza (orden igual a VAR_ORDER) ─────────────────────────
COV = np.array([
    [ 0.047919, -0.022773, -0.000405, -0.011147, -0.005744, -0.011505, -0.004486, -0.081558, -0.016363, -0.022869, -0.019145, -0.000696,  0.006349,  0.007024],
    [-0.022773,  0.023833, -0.000956,  0.007844, -0.003376,  0.000389, -0.006103,  0.069179, -0.000527,  0.004386,  0.000591, -0.005726, -0.004873,  0.000367],
    [-0.000405, -0.000956,  0.003655, -0.000755, -0.000935,  0.000161,  0.000247, -0.000913, -0.000204,  0.000059,  0.000671, -0.001499, -0.000640,  0.000275],
    [-0.011147,  0.007844, -0.000755,  0.021900, -0.001285,  0.000298,  0.000022,  0.054954, -0.000324, -0.000181,  0.000883, -0.000199, -0.014951,  0.001427],
    [-0.005744, -0.003376, -0.000935, -0.001285,  0.032609,  0.000749, -0.001249, -0.004009,  0.002191,  0.001554,  0.004891, -0.001350, -0.000288,  0.001745],
    [-0.011505,  0.000389,  0.000161,  0.000298,  0.000749,  0.061023,  0.009875,  0.011174,  0.000548, -0.000032,  0.001239,  0.001061,  0.000120,  0.003028],
    [-0.004486, -0.006103,  0.000247,  0.000022, -0.001249,  0.009875,  0.034062, -0.005634, -0.000080,  0.001112, -0.000011,  0.001585,  0.000862, -0.004038],
    [-0.081558,  0.069179, -0.000913,  0.054954, -0.004009,  0.011174, -0.005634,  0.794381,  0.000433,  0.014214, -0.019406, -0.007621, -0.044103, -0.002395],
    [-0.016363, -0.000527, -0.000204, -0.000324,  0.002191,  0.000548, -0.000080,  0.000433,  0.037656,  0.016326,  0.016809, -0.002861,  0.000463,  0.002982],
    [-0.022869,  0.004386,  0.000059, -0.000181,  0.001554, -0.000032,  0.001112,  0.014214,  0.016326,  0.069528,  0.016856,  0.005674,  0.000640, -0.058321],
    [-0.019145,  0.000591,  0.000671,  0.000883,  0.004891,  0.001239, -0.000011, -0.019406,  0.016809,  0.016856,  0.050674, -0.001736, -0.000911,  0.001302],
    [-0.000696, -0.005726, -0.001499, -0.000199, -0.001350,  0.001061,  0.001585, -0.007621, -0.002861,  0.005674, -0.001736,  0.064998,  0.000581, -0.062561],
    [ 0.006349, -0.004873, -0.000640, -0.014951, -0.000288,  0.000120,  0.000862, -0.044103,  0.000463,  0.000640, -0.000911,  0.000581,  0.017720, -0.003108],
    [ 0.007024,  0.000367,  0.000275,  0.001427,  0.001745,  0.003028, -0.004038, -0.002395,  0.002982, -0.058321,  0.001302, -0.062561, -0.003108,  0.246351],
])

def logistic(x):
    return 1 / (1 + np.exp(-x))

# ── App ────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Predictor de Riesgo de ACV"

# Colores
NAVY  = "#1B2A4A"
TEAL  = "#0D9488"
RED   = "#E74C3C"
GREEN = "#27AE60"
GRAY  = "#F1F5F9"

app.layout = dbc.Container([

    # ── Header ──────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("Predictor de Riesgo de ACV",
                        style={"color": "white", "margin": 0, "fontWeight": "700"}),
                html.P("Modelo de Regresión Logística — EDA Reducido",
                       style={"color": "#CCFBF1", "margin": 0, "fontSize": "14px"}),
            ], style={
                "background": NAVY, "padding": "20px 30px",
                "borderRadius": "10px", "marginBottom": "25px"
            })
        ])
    ]),

    dbc.Row([
        # ── Panel izquierdo: Inputs ──────────────────────────────────────
        dbc.Col([

            # Variables numéricas
            html.Div([
                html.H5("Variables Numéricas",
                        style={"color": NAVY, "fontWeight": "700", "marginBottom": "15px",
                               "borderBottom": f"2px solid {TEAL}", "paddingBottom": "8px"}),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Edad (años)", style={"fontWeight": "600", "color": NAVY}),
                        dbc.Input(id="input-age", type="number", min=1, max=100,
                                  placeholder="Ej: 55", value=None,
                                  style={"borderRadius": "8px"}),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Glucosa en sangre (mg/dL)", style={"fontWeight": "600", "color": NAVY}),
                        dbc.Input(id="input-glucose", type="number", min=50, max=300,
                                  placeholder="Ej: 105", value=None,
                                  style={"borderRadius": "8px"}),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("IMC (kg/m²)", style={"fontWeight": "600", "color": NAVY}),
                        dbc.Input(id="input-bmi", type="number", min=10, max=60,
                                  placeholder="Ej: 28", value=None,
                                  style={"borderRadius": "8px"}),
                    ], md=4),
                ], className="mb-3"),
            ], style={"background": "white", "padding": "20px",
                      "borderRadius": "10px", "marginBottom": "20px",
                      "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),

            # Variables categóricas
            html.Div([
                html.H5("Variables Categóricas",
                        style={"color": NAVY, "fontWeight": "700", "marginBottom": "15px",
                               "borderBottom": f"2px solid {TEAL}", "paddingBottom": "8px"}),

                dbc.Row([
                    # Condiciones médicas
                    dbc.Col([
                        html.H6("Condiciones médicas",
                                style={"color": TEAL, "fontWeight": "600", "marginBottom": "10px"}),
                        dbc.Checklist(
                            id="check-hypertension",
                            options=[{"label": " Hipertensión arterial", "value": 1}],
                            value=[],
                            style={"marginBottom": "10px"}
                        ),
                        dbc.Checklist(
                            id="check-heart-disease",
                            options=[{"label": " Enfermedad cardíaca preexistente", "value": 1}],
                            value=[],
                        ),
                    ], md=4),

                    # Tipo de trabajo
                    dbc.Col([
                        html.H6("Tipo de trabajo",
                                style={"color": TEAL, "fontWeight": "600", "marginBottom": "10px"}),
                        dbc.RadioItems(
                            id="radio-work",
                            options=[
                                {"label": " Sector privado (referencia)", "value": "private"},
                                {"label": " Sector gubernamental",        "value": "govt"},
                                {"label": " Independiente",               "value": "self"},
                                {"label": " Niño / Sin trabajo",          "value": "children"},
                            ],
                            value="private",
                            labelStyle={"display": "block", "marginBottom": "6px"},
                        ),
                    ], md=4),

                    # Hábito de fumar
                    dbc.Col([
                        html.H6("Hábito de fumar",
                                style={"color": TEAL, "fontWeight": "600", "marginBottom": "10px"}),
                        dbc.RadioItems(
                            id="radio-smoking",
                            options=[
                                {"label": " Nunca fumó (referencia)", "value": "never"},
                                {"label": " Exfumador",               "value": "former"},
                                {"label": " Fumador activo",          "value": "smokes"},
                                {"label": " Desconocido",             "value": "unknown"},
                            ],
                            value="never",
                            labelStyle={"display": "block", "marginBottom": "6px"},
                        ),
                    ], md=4),
                ]),
            ], style={"background": "white", "padding": "20px",
                      "borderRadius": "10px", "marginBottom": "20px",
                      "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),

            # Botón
            dbc.Button("Calcular Riesgo", id="btn-predict", color="primary", size="lg",
                       style={"background": TEAL, "border": "none", "borderRadius": "8px",
                              "width": "100%", "fontWeight": "700", "fontSize": "16px",
                              "padding": "12px"}),

        ], md=7),

        # ── Panel derecho: Resultado ─────────────────────────────────────
        dbc.Col([
            html.Div(id="resultado-panel", children=[
                html.Div([
                    html.Div("🫀", style={"fontSize": "60px", "textAlign": "center",
                                          "marginBottom": "15px"}),
                    html.P("Complete los datos y presione\n'Calcular Riesgo'",
                           style={"color": "#94A3B8", "textAlign": "center",
                                  "fontSize": "15px"}),
                ], style={"padding": "40px 20px"})
            ], style={"background": "white", "borderRadius": "10px", "minHeight": "400px",
                      "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "display": "flex",
                      "alignItems": "center", "justifyContent": "center"}),
        ], md=5),
    ]),

    # ── Footer ──────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Modelo EDA Reducido · Regresión Logística · Variables estandarizadas · "
                   "AUC-ROC = 0.842 · AUC-PR = 0.206",
                   style={"color": "#94A3B8", "fontSize": "12px", "textAlign": "center"}),
        ])
    ]),

], fluid=True, style={"background": "#F8FAFC", "minHeight": "100vh", "padding": "20px"})


# ── Callback ──────────────────────────────────────────────────────────────
@app.callback(
    Output("resultado-panel", "children"),
    Input("btn-predict", "n_clicks"),
    State("input-age",         "value"),
    State("input-glucose",     "value"),
    State("input-bmi",         "value"),
    State("check-hypertension","value"),
    State("check-heart-disease","value"),
    State("radio-work",        "value"),
    State("radio-smoking",     "value"),
    prevent_initial_call=True
)
def predict(n_clicks, age, glucose, bmi, hypertension, heart_disease, work, smoking):

    # Validación
    if any(v is None for v in [age, glucose, bmi]):
        return html.Div([
            html.Div("⚠️", style={"fontSize": "50px", "textAlign": "center"}),
            html.P("Por favor complete todos los campos numéricos.",
                   style={"color": RED, "textAlign": "center", "fontWeight": "600"})
        ], style={"padding": "40px"})

    # Estandarización
    age_std     = (age     - MEANS["age"])     / STDS["age"]
    glucose_std = (glucose - MEANS["glucose"]) / STDS["glucose"]
    bmi_std     = (bmi     - MEANS["bmi"])     / STDS["bmi"]

    # Variables binarias
    htn = 1 if 1 in hypertension else 0
    hd  = 1 if 1 in heart_disease else 0

    wt_govt = 1 if work == "govt"     else 0
    wt_self = 1 if work == "self"     else 0
    wt_chld = 1 if work == "children" else 0

    sm_former  = 1 if smoking == "former"  else 0
    sm_smokes  = 1 if smoking == "smokes"  else 0
    sm_unknown = 1 if smoking == "unknown" else 0

    # Vector x (mismo orden que VAR_ORDER)
    x = np.array([
        1,           # Intercept
        age_std,
        glucose_std,
        bmi_std,
        htn,
        wt_govt,
        wt_self,
        wt_chld,
        sm_former,
        sm_smokes,
        sm_unknown,
        hd,
        age_std * bmi_std,
        hd * sm_smokes,
    ])

    # Predictor lineal
    eta = float(x @ np.array(list(COEF.values())))

    # Error estándar via método delta: SE = sqrt(x^T Sigma x)
    se_eta = float(np.sqrt(x @ COV @ x))

    # IC 95% en escala del predictor lineal
    eta_lo = eta - 1.96 * se_eta
    eta_hi = eta + 1.96 * se_eta

    # Transformar a probabilidad
    prob    = logistic(eta)
    prob_lo = logistic(eta_lo)
    prob_hi = logistic(eta_hi)

    pct    = prob    * 100
    pct_lo = prob_lo * 100
    pct_hi = prob_hi * 100

    # Nivel de riesgo
    if pct < 5:
        color, emoji, nivel, msg = GREEN, "✅", "Riesgo Bajo", "Su perfil no presenta factores de riesgo significativos para ACV."
    elif pct < 15:
        color, emoji, nivel, msg = "#F59E0B", "⚠️", "Riesgo Moderado", "Algunos factores de riesgo presentes. Se recomienda consulta médica preventiva."
    else:
        color, emoji, nivel, msg = RED, "🚨", "Riesgo Alto", "Presencia de múltiples factores de riesgo. Se recomienda consulta médica urgente."

    # Barra de progreso
    bar_pct = min(pct * 2, 100)  # escalar para mejor visualización

    # Factores activos
    factores = []
    if age > 60:        factores.append("Edad avanzada")
    if htn:             factores.append("Hipertensión")
    if hd and sm_smokes:factores.append("Enf. cardíaca + fumador activo")
    elif hd:            factores.append("Enfermedad cardíaca")
    if glucose > 140:   factores.append("Glucosa elevada")
    if bmi > 30:        factores.append("IMC elevado (obesidad)")

    return html.Div([
        # Emoji y nivel
        html.Div(emoji, style={"fontSize": "60px", "textAlign": "center", "marginBottom": "10px"}),
        html.H4(nivel, style={"color": color, "textAlign": "center", "fontWeight": "700",
                               "marginBottom": "5px"}),

        # Probabilidad grande
        html.Div([
            html.Span(f"{pct:.1f}%",
                      style={"fontSize": "52px", "fontWeight": "800", "color": color}),
            html.Span("probabilidad estimada de ACV",
                      style={"fontSize": "13px", "color": "#64748B", "display": "block",
                             "textAlign": "center"}),
        ], style={"textAlign": "center", "marginBottom": "10px"}),

        # IC 95%
        html.Div([
            html.Span("IC 95%: ", style={"fontWeight": "600", "color": NAVY, "fontSize": "13px"}),
            html.Span(f"[{pct_lo:.1f}% — {pct_hi:.1f}%]",
                      style={"color": color, "fontWeight": "700", "fontSize": "14px"}),
        ], style={"textAlign": "center", "marginBottom": "15px",
                  "background": GRAY, "borderRadius": "8px", "padding": "8px 15px",
                  "display": "inline-block", "width": "100%"}),

        # Barra de riesgo
        html.Div([
            html.Div(style={
                "height": "12px", "borderRadius": "6px",
                "background": f"linear-gradient(to right, {GREEN}, #F59E0B, {RED})",
                "marginBottom": "4px", "position": "relative"
            }),
            html.Div(style={
                "width": "12px", "height": "20px", "background": NAVY,
                "borderRadius": "3px", "position": "relative",
                "left": f"calc({min(pct*2, 98):.1f}% - 6px)",
                "top": "-16px"
            }),
        ], style={"marginBottom": "15px", "padding": "0 5px"}),

        # Mensaje
        html.P(msg, style={"textAlign": "center", "color": "#475569",
                            "fontSize": "13px", "marginBottom": "15px"}),

        # Factores de riesgo activos
        html.Div([
            html.P("Factores de riesgo identificados:",
                   style={"fontWeight": "600", "color": NAVY, "marginBottom": "8px",
                          "fontSize": "13px"}),
            html.Div([
                html.Span(f, style={
                    "background": "#FEF3C7", "color": "#92400E", "borderRadius": "12px",
                    "padding": "3px 10px", "fontSize": "12px", "marginRight": "5px",
                    "marginBottom": "5px", "display": "inline-block"
                }) for f in factores
            ] if factores else [html.Span("Ninguno identificado",
                                           style={"color": GREEN, "fontSize": "13px"})])
        ], style={"background": GRAY, "borderRadius": "8px", "padding": "12px",
                  "marginBottom": "10px"}),

        # Disclaimer
        html.P("⚕️ Esta herramienta es solo orientativa y no reemplaza el diagnóstico médico.",
               style={"fontSize": "11px", "color": "#94A3B8", "textAlign": "center",
                      "fontStyle": "italic"}),

    ], style={"padding": "20px"})


if __name__ == "__main__":
    app.run(debug=True, port=8050)

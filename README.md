# Recomendador HORECA (Two-Tower + Re-Ranking)

Proyecto end-to-end siguiendo CRISP-DM para un sistema de recomendación de productos HORECA, orientado a:
1) Maximizar la probabilidad de compra por cliente
2) Reducir productos de baja rotación
3) Priorizar productos próximos a vencer

## 1. Business Understanding
- **Objetivo comercial:** incrementar conversión y margen, a la vez que se reduce inventario de baja rotación y se mitigan mermas por caducidad.
- **KPIs principales:** Recall@K, NDCG@K, uplift en tasa de recompra, reducción de inventario lento, tasa de venta de productos próximos a vencer.
- **Estrategia:** modelo de Deep Learning Two-Tower con re-ranking de negocio:  
  `score_final = score_dl + α * indice_baja_rotacion + β * indice_proximo_vencer`

## 2. Data Understanding
Fuentes y variables disponibles:
- **Clientes:** `cliente_id`, `rubro_cliente`
- **Productos:** `producto_id`, `categoria_producto`, `precio_unitario`, `costo_unitario`, `stock`, `fecha_ingreso_catalogo`, `fecha_min_caducidad`
- **Ventas:** `venta_id`, `cliente_id`, `fecha_venta`, `canal_venta`, `monto_total`
- **Detalle_Venta:** `venta_id`, `producto_id`, `cantidad_producto`, `subtotal`, `descuento_aplicado`

## 3. Data Preparation
- Generación de dataset a nivel **(cliente, producto, contexto)**.
- Features derivadas:
  - `frecuencia_compra`, `ticket_promedio`.
  - `mes`, `semana_anio`, `cerca_feriado_7d`.
  - `dias_para_vencer`, `indice_proximo_vencer` (0–1).
  - `rotacion_producto`, `indice_baja_rotacion` (0–1).
- **Split temporal** por ventana de entrenamiento/validación/test y **label** `label_compra`.
- **Negative sampling** por cliente con productos no comprados.

Scripts:
```bash
python data/generate_dummy_data.py
python -m src.features.build_dataset
```

## 4. Modeling
- **Arquitectura:** Two-Tower (User Tower + Item Tower).
- **Loss:** BCE (por defecto) o BPR.
- **Entrenamiento:** `python -m src.models.train`

### Algoritmo(s) de Deep Learning y cómo se aplican
Se utiliza un modelo **Two-Tower** de Deep Learning para representar usuarios y productos en el mismo espacio latente. La idea central es aprender dos funciones (torres) que transforman features de usuario y de producto en embeddings comparables. Esto permite:
- **Maximizar probabilidad de compra:** la similitud entre embeddings (producto y cliente) actúa como score de afinidad.
- **Escalabilidad:** los embeddings de productos pueden precomputarse y luego compararse rápidamente con usuarios.

**Detalle del flujo:**
1. **Tower de usuario:** combina embeddings de variables categóricas (cliente_id, rubro_cliente, canal_venta) con variables numéricas (frecuencia_compra, ticket_promedio, contexto temporal).
2. **Tower de producto:** combina embeddings de variables categóricas (producto_id, categoria_producto) con variables numéricas (precio, stock, rotación, caducidad).
3. **Score DL:** producto punto entre ambos embeddings (afinidad).
4. **Re-ranking de negocio:** se ajusta el score usando índices de baja rotación y proximidad a vencimiento:  
   `score_final = score_dl + α * indice_baja_rotacion + β * indice_proximo_vencer`.

El objetivo es equilibrar precisión predictiva (probabilidad de compra) con objetivos operativos (salida de inventario lento y próximos a vencer).

## 5. Evaluation
Métricas de ranking:
- `Recall@K`
- `NDCG@K`

```bash
python -m src.evaluation.evaluate
```

## 6. Deployment
- API con FastAPI: `api/app.py`
- Función de inferencia: `src/inference/recommend.py`

```bash
uvicorn api.app:app --reload
```

## Estructura del proyecto
```
PT1/
├── api/
├── data/
│   ├── clientes.csv
│   ├── productos.csv
│   ├── ventas.csv
│   ├── detalle_venta.csv
│   └── processed/
├── notebooks/
├── src/
│   ├── evaluation/
│   ├── features/
│   ├── inference/
│   └── models/
└── README.md
```

## Ejemplo de uso
```python
from src.inference.recommend import recomendar

recomendaciones = recomendar("C001", top_k=5)
print(recomendaciones)
```

## Requisitos
- Python 3.10+
- pandas
- numpy
- torch
- fastapi
- uvicorn

Instalación sugerida:
```bash
pip install -r requirements.txt
```

## Flujo end-to-end
```bash
python data/generate_dummy_data.py
python -m src.features.build_dataset
python -m src.models.train
python -m src.evaluation.evaluate
python -m src.inference.recommend
```

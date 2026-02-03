import csv
import random
from datetime import datetime, timedelta

random.seed(42)

clientes = [
    {"cliente_id": f"C{idx:03d}", "rubro_cliente": random.choice(["restaurante", "hotel", "catering", "cafeteria"])}
    for idx in range(1, 11)
]

productos = []
for idx in range(1, 16):
    productos.append(
        {
            "producto_id": f"P{idx:03d}",
            "categoria_producto": random.choice(["bebidas", "lacteos", "carnes", "vegetales", "panaderia"]),
            "precio_unitario": round(random.uniform(1.0, 50.0), 2),
            "costo_unitario": round(random.uniform(0.5, 30.0), 2),
            "stock": random.randint(10, 200),
            "fecha_ingreso_catalogo": (datetime.today() - timedelta(days=random.randint(30, 365))).date().isoformat(),
            "fecha_min_caducidad": (datetime.today() + timedelta(days=random.randint(5, 120))).date().isoformat(),
        }
    )

ventas = []
venta_detalle = []
venta_id = 1
for cliente in clientes:
    n_ventas = random.randint(4, 8)
    for _ in range(n_ventas):
        fecha_venta = datetime.today() - timedelta(days=random.randint(1, 90))
        canal_venta = random.choice(["online", "fuerza_ventas", "distribuidor"])
        productos_compra = random.sample([p["producto_id"] for p in productos], 3)
        monto_total = 0.0
        for prod_id in productos_compra:
            cantidad = random.randint(1, 8)
            precio = next(p["precio_unitario"] for p in productos if p["producto_id"] == prod_id)
            subtotal = round(cantidad * precio, 2)
            descuento = round(subtotal * random.choice([0, 0.05, 0.1]), 2)
            monto_total += subtotal - descuento
            venta_detalle.append(
                {
                    "venta_id": f"V{venta_id:04d}",
                    "producto_id": prod_id,
                    "cantidad_producto": cantidad,
                    "subtotal": subtotal,
                    "descuento_aplicado": descuento,
                }
            )
        ventas.append(
            {
                "venta_id": f"V{venta_id:04d}",
                "cliente_id": cliente["cliente_id"],
                "fecha_venta": fecha_venta.date().isoformat(),
                "canal_venta": canal_venta,
                "monto_total": round(monto_total, 2),
            }
        )
        venta_id += 1


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


write_csv("/workspace/PT1/data/clientes.csv", clientes, ["cliente_id", "rubro_cliente"])
write_csv(
    "/workspace/PT1/data/productos.csv",
    productos,
    [
        "producto_id",
        "categoria_producto",
        "precio_unitario",
        "costo_unitario",
        "stock",
        "fecha_ingreso_catalogo",
        "fecha_min_caducidad",
    ],
)
write_csv(
    "/workspace/PT1/data/ventas.csv",
    ventas,
    ["venta_id", "cliente_id", "fecha_venta", "canal_venta", "monto_total"],
)
write_csv(
    "/workspace/PT1/data/detalle_venta.csv",
    venta_detalle,
    ["venta_id", "producto_id", "cantidad_producto", "subtotal", "descuento_aplicado"],
)

print("Dummy data generated in /workspace/PT1/data")

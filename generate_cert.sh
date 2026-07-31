#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_cert.sh — Generate self-signed SSL certificate untuk Flask HTTPS
# Jalankan di Raspberry Pi dari root direktori proyek:
#   chmod +x generate_cert.sh && ./generate_cert.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

CERT_DIR="certs"
KEY_FILE="$CERT_DIR/server.key"
CERT_FILE="$CERT_DIR/server.crt"
DAYS=365

echo "[INFO] Membuat direktori $CERT_DIR ..."
mkdir -p "$CERT_DIR"

echo "[INFO] Generate RSA 2048-bit private key dan self-signed certificate ..."
openssl req -x509 \
  -newkey rsa:2048 \
  -keyout "$KEY_FILE" \
  -out    "$CERT_FILE" \
  -days   "$DAYS" \
  -nodes \
  -subj   "/C=ID/ST=Local/L=Local/O=AccessControl/CN=access-control-local"

# Batasi permission file key (hanya owner yang bisa baca)
chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo ""
echo "✅ Sertifikat berhasil dibuat:"
echo "   Key : $KEY_FILE"
echo "   Cert: $CERT_FILE"
echo ""
echo "Sekarang aktifkan HTTPS di config.py:"
echo "   WEB_USE_SSL = True"
echo ""
echo "Informasi sertifikat:"
openssl x509 -in "$CERT_FILE" -noout -subject -dates

#!/usr/bin/env bash
set -e
# Wait for mongod to be ready
until mongo --quiet --eval "db.adminCommand('ping')"; do
  sleep 1
done

# Import each JSON file as its own collection
# Adjust collection names as you like
mongoimport \
  --db mydatabase \
  --collection property_models \
  --file /docker-entrypoint-initdb.d/models/property_models.json \
  --jsonArray

# If you have multiple files in metamodels/ and data_models/:
for f in /docker-entrypoint-initdb.d/models/metamodels/*.json; do
  mongoimport --db mydatabase --collection metamodels --file "$f" --jsonArray
done

for f in /docker-entrypoint-initdb.d/models/data_models/*.json; do
  mongoimport --db mydatabase --collection data_models --file "$f" --jsonArray
done

export $(grep -v '^#' .env | xargs)
./mvnw test
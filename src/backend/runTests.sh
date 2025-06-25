export $(grep -v '^#' .env | xargs)
./mvnw clean
./mvnw compile test-compile
./mvnw test
./mvnw jacoco:report
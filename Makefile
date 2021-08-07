main.wasm : ./cmd/bayesplay/main.go ./pkg/distributions/distributions.go ./pkg/bayesfactor/bayesfactor.go
	GOOS=js GOARCH=wasm go build -o dist/main.wasm cmd/bayesplay/main.go

tests :
	go test pkg/distributions/distributions.go pkg/distributions/distributions_test.go
	go test pkg/bayesfactor/bayesfactor.go pkg/bayesfactor/bayesfactor_test.go

clean : FORCE
	rm dist/main.wasm

FORCE :

all :
	make clean
	make test
	make main.wasm

distribution :
	make main.wasm
	cp dist/main.wasm ../bayesplay-front-end/build/main.wasm
	cp dist/main.wasm ../bayesplay-front-end/public/main.wasm	

main.wasm : ./cmd/bayesplay/main.go ./pkg/distributions/distributions.go ./pkg/bayesfactor/bayesfactor.go
	GOOS=js GOARCH=wasm go build -o main.wasm cmd/bayesplay/main.go

test :
	go test pkg/distributions/distributions.go pkg/distributions/distributions_test.go
	go test pkg/bayesfactor/bayesfactor.go pkg/bayesfactor/bayesfactor_test.go

clean : FORCE
	rm main.wasm

FORCE :

all :
	make clean
	make test
	make main.wasm



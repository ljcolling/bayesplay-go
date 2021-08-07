// +build js wasm
module bayesplay

go 1.16

require pkg/bayesfactor v1.0.0

replace pkg/bayesfactor => ../../pkg/bayesfactor

require (
	github.com/google/go-cmp v0.5.6
	pkg/distributions v1.0.0
)

replace pkg/distributions => ../../pkg/distributions

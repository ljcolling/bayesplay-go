module bayesfactor

go 1.16

require (
	github.com/google/go-cmp v0.5.6
	golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1 // indirect
)

require (
	pkg/distributions v1.0.0
	scientificgo.org/special v0.0.0 // indirect
)

replace pkg/distributions => ../distributions

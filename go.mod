module bayesplay-go

go 1.16

require pkg/bayesfactor v1.0.0
replace pkg/bayesfactor => ./pkg/bayesfactor
require	pkg/distributions v1.0.0
replace pkg/distributions => ./pkg/distributions

require scientificgo.org/special v0.0.0 // indirect

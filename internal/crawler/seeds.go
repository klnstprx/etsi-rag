package crawler

import "etsi-rag/internal/domain"

// DefaultSeeds returns the default crawl sources for ETSI Informatica.
func DefaultSeeds() []domain.CrawlSource {
	return []domain.CrawlSource{
		{Name: "Grado en Ingeniería Informática", SeedURL: "https://www.uma.es/grado-en-ingenieria-informatica/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Grado en Ingeniería del Software", SeedURL: "https://www.uma.es/grado-en-ingenieria-del-software/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Grado en Ingeniería de Computadores", SeedURL: "https://www.uma.es/grado-en-ingenieria-de-computadores/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Grado en Ingeniería de la Salud", SeedURL: "https://www.uma.es/grado-en-ingenieria-de-la-salud/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Doble Grado Informática-Matemáticas", SeedURL: "https://www.uma.es/informatica-matematicas/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Grado en Ciberseguridad e IA", SeedURL: "https://www.uma.es/grado-en-ciberseguridad-e-inteligencia-artificial/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Máster en Ingeniería Informática", SeedURL: "https://www.uma.es/master-en-ingenieria-informatica/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Máster en Ingeniería del Software e IA", SeedURL: "https://www.uma.es/master-en-ingenieria-del-software-e-inteligencia-artificial/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Máster en Ciberseguridad", SeedURL: "https://www.uma.es/master-en-ciberseguridad/", Category: "degree", Language: "es", Enabled: true},
		{Name: "Doctorado en Informática", SeedURL: "https://www.uma.es/doctorado-informatica/", Category: "degree", Language: "es", Enabled: true},
		{Name: "ETSI Informática", SeedURL: "https://www.uma.es/etsi-informatica/", Category: "general", Language: "es", Enabled: true},
	}
}

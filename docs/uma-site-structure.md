# UMA Site Structure
<!-- packages: internal/crawler -->
<!-- related: docs/crawler.md -->

Reference for the Universidad de Malaga website structure, URL patterns, and content types relevant to the crawler.

## Overview

- **Static HTML** site (Plone CMS, jQuery, server-rendered)
- No sitemap, no API, no robots.txt — must follow links from seed pages
- **Two domains**: `www.uma.es` (HTML pages), `sara.uma.es` (guia docente PDFs)
- Content is predominantly in Spanish

## URL Patterns

```
Seed page (e.g. /grado-en-ingenieria-informatica/)
  +-- /{slug}/cms/menu/{section}/{subsection}/     <- CMS menu pages
  +-- /{slug}/info/{numeric-id}/{page-slug}/       <- Info pages
  +-- /centers/subject/{degree_id}/{course_id}     <- Course pages (JS-rendered)
  |     +-- sara.uma.es/ht/{year}/ProgramasAsignaturas_Titulacion_{id}_AsigUMA_{id}.pdf
  +-- /media/files/{filename}.pdf                  <- Memorias, BOE docs
  +-- /etsi-informatica/...                        <- School-level shared pages
```

## Link Chain to Guia Docente PDFs

1. Seed -> `programacion-docente` page (lists all courses as links)
2. Each course -> `/centers/subject/{degree_id}/{course_id}` (JS-rendered, needs go-rod)
3. Each subject page -> exactly one PDF on `sara.uma.es`

Note: `plan-de-estudios` pages list courses as **plain text** (no links) — only `programacion-docente` has clickable subject links.

## Seed URLs

| Seed | URL |
|------|-----|
| Grado Informatica | `/grado-en-ingenieria-informatica/` |
| Grado Software | `/grado-en-ingenieria-del-software/` |
| Grado Computadores | `/grado-en-ingenieria-de-computadores/` |
| Grado Salud | `/grado-en-ingenieria-de-la-salud/` |
| Doble Grado Mate+Info | `/informatica-matematicas/` |
| Grado Ciberseguridad+IA | `/grado-en-ciberseguridad-e-inteligencia-artificial/` |
| Master Informatica | `/master-en-ingenieria-informatica/` |
| Master Software+IA | `/master-en-ingenieria-del-software-e-inteligencia-artificial/` |
| Master Ciberseguridad | `/master-en-ciberseguridad/` |
| Doctorado Informatica | `/doctorado-informatica/` |
| ETSI General | `/etsi-informatica/` |

All URLs relative to `https://www.uma.es`.

## Content Types

| Type | Source | Content |
|------|--------|---------|
| Degree pages | HTML | Overview, ECTS, specializations, competencies |
| CMS menu pages | HTML | Procedures, admissions, enrollment info |
| Course pages | HTML (JS) | Individual course info, link to guia docente |
| Guias docentes | PDF | Syllabus, evaluation criteria, bibliography, tutoring hours |
| Memorias de verificacion | PDF | Government-mandated degree architecture docs |
| Academic calendars | PDF | Start dates, holidays, exam windows |

## CMS Behavior

- Plone-based CMS with Open Graph metadata (`og:title`, `og:url`, `og:description`)
- `lang` attribute on `<html>` tag for language detection
- Left sidebars contain relevant local navigation; right sidebars are mostly generic banners
- Global boilerplate: nav headers, institutional footer (seals, social links) — stripped by go-readability

## Scope Boundaries

- **In scope**: ETSI Informatica degrees, masters, doctoral, school-level pages
- **Out of scope**: external microsites (`muii.informatica.uma.es`), news (`/noticias/`), other faculties
- News excluded from batch crawl — time-sensitive, handled via live agent tool

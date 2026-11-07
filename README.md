# QNE & QNE Datacenter — GF(2^128)

**Machine de référence : AMD Ryzen 9 9950X3D, 16 cœurs / 32 threads.**

Campagne du 7 novembre 2026, Ubuntu WSL, C++20, Node-API, VPCLMULQDQ / AVX-512. Banques 4 Kio désactivées.

Ce dépôt contient **QNE GF128 V13**, moteur de multiplications indépendantes, et **QNE Datacenter GF128 V9**, moteur d'agrégation de 10 régions à threads persistants.

## Résultats mesurés

| Moteur | Débit | Protocole |
|---|---:|---|
| QNE V8 historique | **1 517 908 319 mul/s** | 4 194 304 opérations, 16 threads |
| QNE V13 autotuning | 1 521 007 346 mul/s | Réglage, 7 passages |
| **QNE V13 confirmé** | **1 489 531 773 mul/s** | 4 194 304 opérations, 13 passages alternés, nouvelles données |
| **QNE Datacenter V9** | **88 659 448 requêtes/s** | 32 768 requêtes, 10 régions, 8 threads |

Le record de réglage n'est pas un record confirmé. Une requête Datacenter n'est pas une multiplication GF128 unique. Les multiplications GF128 ne sont pas des FLOPS.

## Comparaison Apple — indicative

| Plateforme | Résultat | Conditions |
|---|---:|---|
| QNE V13 / Ryzen 9950X3D | **1 489,53 millions GF128/s** | Débit agrégé multithread confirmé |
| Apple M3 Max / Hekate Math | **0,90 ns par multiplication**, environ 1 111 millions/s | PMULL plat, pipeliné ; protocole différent |
| Référence GHASH historique QNE | 354,48 millions/s | Matériel et protocole non équivalents |

Source Apple : [Hekate Math](https://github.com/oumuamua-labs/hekate-math).

**Ces résultats ne permettent pas de conclure qu'un processeur est globalement plus rapide qu'un autre.**

## Tableau des processeurs

Catégories définies par le nombre de cœurs, **pas par un classement GF128**.

| Catégorie | Processeur | Cœurs / threads | QNE testé |
|---|---|---:|---|
| Moins de cœurs | AMD Ryzen 7 9700X | 8 / 16 | Non |
| Même nombre de cœurs | AMD Ryzen 9 9950X | 16 / 32 | Non |
| **Référence** | **AMD Ryzen 9 9950X3D** | **16 / 32** | **Oui** |
| Plus de cœurs | AMD Threadripper 9980X | 64 / 128 | Non |
| Plus de cœurs | AMD EPYC 9965 | 192 / 384 | Non |
| Architecture ARM | Apple M3 Max | 14 ou 16 cœurs CPU | QNE non porté |

## Architecture

- `qne/core/reconstruct.js` : référence mathématique.
- `qne/gf128-v13/` : accélérateur natif QNE V13.
- `qne-datacenter/gf128-v9/` : accélérateur Datacenter V9.
- `benchmarks/resultats.json` : relevé de campagne.
- `scripts/build.sh` : compilation des deux modules.

## Compilation

Environnement Linux x86-64 compatible AVX-512 / VPCLMUL, Node.js avec en-têtes Node-API, g++ C++20 et pthreads.

```bash
bash scripts/build.sh
```

La compilation seule ne reproduit pas les mesures : les tailles des lots, les données, les threads et les conditions de test doivent également correspondre.

## Intégrité et limites

Les résultats proviennent des journaux de benchmark exécutés sur le Ryzen. Ils ne constituent ni un audit cryptographique indépendant ni une certification interprocesseurs.

Les sources originales ne sont pas modifiées par la publication. Les binaires compilés et les secrets sont exclus du dépôt.

Aucune licence ouverte n'est accordée sans fichier de licence explicite.

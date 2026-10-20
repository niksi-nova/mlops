# Delhi Bus GPS - Raw Data Inventory

Source: `data.source_url` in `params.yaml` (https://github.com/bus-data/bus-data.github.io/raw/master/asset/dataset.zip), downloaded and unzipped into `data/raw/delhi_gps/` (43.5063 MB on disk, 184 files).

## Layout

Each file is one day of node-crossing events for one bus route: `Month_<m>/<route>/Day<d>.xlsx`.

## Per-file summary

184 files, all `.xlsx`, identical schema. Size ranges 0.026-0.2676 MB per file, rows per file range 794-10013 (min: 794, median: 3597.5, max: 10013).

| file | format | size (MB) | rows | route | month | day |
|---|---|---|---|---|---|---|
| Month_1/425/Day1.xlsx | xlsx | 0.0638 | 2227 | 425 | 1 | 1 |
| Month_1/425/Day2.xlsx | xlsx | 0.0714 | 2517 | 425 | 1 | 2 |
| Month_1/425/Day3.xlsx | xlsx | 0.0569 | 1970 | 425 | 1 | 3 |
| Month_1/425/Day4.xlsx | xlsx | 0.059 | 2047 | 425 | 1 | 4 |
| Month_1/425/Day5.xlsx | xlsx | 0.0647 | 2272 | 425 | 1 | 5 |
| Month_1/425/Day6.xlsx | xlsx | 0.0581 | 2006 | 425 | 1 | 6 |
| Month_1/425/Day7.xlsx | xlsx | 0.0617 | 2140 | 425 | 1 | 7 |
| Month_1/425/Day8.xlsx | xlsx | 0.0648 | 2268 | 425 | 1 | 8 |
| Month_1/425/Day9.xlsx | xlsx | 0.0534 | 1836 | 425 | 1 | 9 |
| Month_1/425/Day10.xlsx | xlsx | 0.0514 | 1752 | 425 | 1 | 10 |
| Month_1/425/Day11.xlsx | xlsx | 0.0474 | 1611 | 425 | 1 | 11 |
| Month_1/425/Day12.xlsx | xlsx | 0.059 | 2056 | 425 | 1 | 12 |
| Month_1/425/Day13.xlsx | xlsx | 0.0613 | 2129 | 425 | 1 | 13 |
| Month_1/425/Day14.xlsx | xlsx | 0.0559 | 1926 | 425 | 1 | 14 |
| Month_1/425/Day15.xlsx | xlsx | 0.0597 | 2074 | 425 | 1 | 15 |
| Month_1/425/Day16.xlsx | xlsx | 0.0486 | 1656 | 425 | 1 | 16 |
| Month_1/425/Day17.xlsx | xlsx | 0.0514 | 1752 | 425 | 1 | 17 |
| Month_1/425/Day18.xlsx | xlsx | 0.0561 | 1936 | 425 | 1 | 18 |
| Month_1/425/Day19.xlsx | xlsx | 0.0486 | 1652 | 425 | 1 | 19 |
| Month_1/425/Day20.xlsx | xlsx | 0.0658 | 2290 | 425 | 1 | 20 |
| Month_1/425/Day21.xlsx | xlsx | 0.0663 | 2318 | 425 | 1 | 21 |
| Month_1/425/Day22.xlsx | xlsx | 0.0438 | 1473 | 425 | 1 | 22 |
| Month_1/425/Day23.xlsx | xlsx | 0.0583 | 2027 | 425 | 1 | 23 |
| Month_1/425/Day24.xlsx | xlsx | 0.0603 | 2091 | 425 | 1 | 24 |
| Month_1/425/Day25.xlsx | xlsx | 0.0568 | 1953 | 425 | 1 | 25 |
| Month_1/425/Day26.xlsx | xlsx | 0.0634 | 2210 | 425 | 1 | 26 |
| Month_1/425/Day27.xlsx | xlsx | 0.0528 | 1812 | 425 | 1 | 27 |
| Month_1/425/Day28.xlsx | xlsx | 0.0616 | 2142 | 425 | 1 | 28 |
| Month_1/425/Day29.xlsx | xlsx | 0.0653 | 2284 | 425 | 1 | 29 |
| Month_1/425/Day30.xlsx | xlsx | 0.0551 | 1901 | 425 | 1 | 30 |
| Month_1/425/Day31.xlsx | xlsx | 0.0657 | 2294 | 425 | 1 | 31 |
| Month_1/534/Day1.xlsx | xlsx | 0.2676 | 10013 | 534 | 1 | 1 |
| Month_1/534/Day2.xlsx | xlsx | 0.2526 | 9451 | 534 | 1 | 2 |
| Month_1/534/Day3.xlsx | xlsx | 0.2401 | 8966 | 534 | 1 | 3 |
| Month_1/534/Day4.xlsx | xlsx | 0.2445 | 9160 | 534 | 1 | 4 |
| Month_1/534/Day5.xlsx | xlsx | 0.2387 | 8920 | 534 | 1 | 5 |
| Month_1/534/Day6.xlsx | xlsx | 0.2223 | 8286 | 534 | 1 | 6 |
| Month_1/534/Day7.xlsx | xlsx | 0.2158 | 8011 | 534 | 1 | 7 |
| Month_1/534/Day8.xlsx | xlsx | 0.2612 | 9815 | 534 | 1 | 8 |
| Month_1/534/Day9.xlsx | xlsx | 0.2287 | 8516 | 534 | 1 | 9 |
| Month_1/534/Day10.xlsx | xlsx | 0.2261 | 8463 | 534 | 1 | 10 |
| Month_1/534/Day11.xlsx | xlsx | 0.2274 | 8486 | 534 | 1 | 11 |
| Month_1/534/Day12.xlsx | xlsx | 0.2189 | 8161 | 534 | 1 | 12 |
| Month_1/534/Day13.xlsx | xlsx | 0.1804 | 6655 | 534 | 1 | 13 |
| Month_1/534/Day14.xlsx | xlsx | 0.2314 | 8621 | 534 | 1 | 14 |
| Month_1/534/Day15.xlsx | xlsx | 0.2235 | 8340 | 534 | 1 | 15 |
| Month_1/534/Day16.xlsx | xlsx | 0.2064 | 7676 | 534 | 1 | 16 |
| Month_1/534/Day17.xlsx | xlsx | 0.1601 | 5908 | 534 | 1 | 17 |
| Month_1/534/Day18.xlsx | xlsx | 0.1802 | 6686 | 534 | 1 | 18 |
| Month_1/534/Day19.xlsx | xlsx | 0.1268 | 4678 | 534 | 1 | 19 |
| Month_1/534/Day20.xlsx | xlsx | 0.2061 | 7693 | 534 | 1 | 20 |
| Month_1/534/Day21.xlsx | xlsx | 0.22 | 8189 | 534 | 1 | 21 |
| Month_1/534/Day22.xlsx | xlsx | 0.2512 | 9410 | 534 | 1 | 22 |
| Month_1/534/Day23.xlsx | xlsx | 0.2072 | 7705 | 534 | 1 | 23 |
| Month_1/534/Day24.xlsx | xlsx | 0.2015 | 7490 | 534 | 1 | 24 |
| Month_1/534/Day25.xlsx | xlsx | 0.2184 | 8145 | 534 | 1 | 25 |
| Month_1/534/Day26.xlsx | xlsx | 0.2079 | 7719 | 534 | 1 | 26 |
| Month_1/534/Day27.xlsx | xlsx | 0.2347 | 8764 | 534 | 1 | 27 |
| Month_1/534/Day28.xlsx | xlsx | 0.2505 | 9363 | 534 | 1 | 28 |
| Month_1/534/Day29.xlsx | xlsx | 0.1487 | 5481 | 534 | 1 | 29 |
| Month_1/534/Day30.xlsx | xlsx | 0.1995 | 7400 | 534 | 1 | 30 |
| Month_1/534/Day31.xlsx | xlsx | 0.2129 | 7914 | 534 | 1 | 31 |
| Month_2/425/Day1.xlsx | xlsx | 0.064 | 2237 | 425 | 2 | 1 |
| Month_2/425/Day2.xlsx | xlsx | 0.0655 | 2298 | 425 | 2 | 2 |
| Month_2/425/Day3.xlsx | xlsx | 0.0615 | 2137 | 425 | 2 | 3 |
| Month_2/425/Day4.xlsx | xlsx | 0.0695 | 2438 | 425 | 2 | 4 |
| Month_2/425/Day5.xlsx | xlsx | 0.0627 | 2188 | 425 | 2 | 5 |
| Month_2/425/Day6.xlsx | xlsx | 0.0604 | 2092 | 425 | 2 | 6 |
| Month_2/425/Day7.xlsx | xlsx | 0.0583 | 2022 | 425 | 2 | 7 |
| Month_2/425/Day8.xlsx | xlsx | 0.0477 | 1624 | 425 | 2 | 8 |
| Month_2/425/Day9.xlsx | xlsx | 0.071 | 2501 | 425 | 2 | 9 |
| Month_2/425/Day10.xlsx | xlsx | 0.0639 | 2227 | 425 | 2 | 10 |
| Month_2/425/Day11.xlsx | xlsx | 0.0564 | 1945 | 425 | 2 | 11 |
| Month_2/425/Day12.xlsx | xlsx | 0.0541 | 1863 | 425 | 2 | 12 |
| Month_2/425/Day13.xlsx | xlsx | 0.0584 | 2031 | 425 | 2 | 13 |
| Month_2/425/Day14.xlsx | xlsx | 0.0569 | 1977 | 425 | 2 | 14 |
| Month_2/425/Day15.xlsx | xlsx | 0.0584 | 2015 | 425 | 2 | 15 |
| Month_2/425/Day16.xlsx | xlsx | 0.0646 | 2258 | 425 | 2 | 16 |
| Month_2/425/Day17.xlsx | xlsx | 0.0432 | 1452 | 425 | 2 | 17 |
| Month_2/425/Day18.xlsx | xlsx | 0.0586 | 2022 | 425 | 2 | 18 |
| Month_2/425/Day19.xlsx | xlsx | 0.026 | 794 | 425 | 2 | 19 |
| Month_2/425/Day20.xlsx | xlsx | 0.0499 | 1703 | 425 | 2 | 20 |
| Month_2/425/Day21.xlsx | xlsx | 0.0587 | 2028 | 425 | 2 | 21 |
| Month_2/425/Day22.xlsx | xlsx | 0.0603 | 2085 | 425 | 2 | 22 |
| Month_2/425/Day23.xlsx | xlsx | 0.0608 | 2116 | 425 | 2 | 23 |
| Month_2/425/Day24.xlsx | xlsx | 0.06 | 2077 | 425 | 2 | 24 |
| Month_2/425/Day25.xlsx | xlsx | 0.0577 | 2003 | 425 | 2 | 25 |
| Month_2/425/Day26.xlsx | xlsx | 0.057 | 1966 | 425 | 2 | 26 |
| Month_2/425/Day27.xlsx | xlsx | 0.059 | 2043 | 425 | 2 | 27 |
| Month_2/425/Day28.xlsx | xlsx | 0.0608 | 2100 | 425 | 2 | 28 |
| Month_2/425/Day29.xlsx | xlsx | 0.0585 | 2025 | 425 | 2 | 29 |
| Month_2/425/Day30.xlsx | xlsx | 0.0667 | 2338 | 425 | 2 | 30 |
| Month_2/534/Day1.xlsx | xlsx | 0.2242 | 8357 | 534 | 2 | 1 |
| Month_2/534/Day2.xlsx | xlsx | 0.2002 | 7404 | 534 | 2 | 2 |
| Month_2/534/Day3.xlsx | xlsx | 0.213 | 7931 | 534 | 2 | 3 |
| Month_2/534/Day4.xlsx | xlsx | 0.2164 | 8051 | 534 | 2 | 4 |
| Month_2/534/Day5.xlsx | xlsx | 0.2385 | 8932 | 534 | 2 | 5 |
| Month_2/534/Day6.xlsx | xlsx | 0.2076 | 7722 | 534 | 2 | 6 |
| Month_2/534/Day7.xlsx | xlsx | 0.2076 | 7721 | 534 | 2 | 7 |
| Month_2/534/Day8.xlsx | xlsx | 0.2004 | 7460 | 534 | 2 | 8 |
| Month_2/534/Day9.xlsx | xlsx | 0.2411 | 8985 | 534 | 2 | 9 |
| Month_2/534/Day10.xlsx | xlsx | 0.2367 | 8840 | 534 | 2 | 10 |
| Month_2/534/Day11.xlsx | xlsx | 0.2323 | 8659 | 534 | 2 | 11 |
| Month_2/534/Day12.xlsx | xlsx | 0.2343 | 8735 | 534 | 2 | 12 |
| Month_2/534/Day13.xlsx | xlsx | 0.2453 | 9140 | 534 | 2 | 13 |
| Month_2/534/Day14.xlsx | xlsx | 0.2281 | 8512 | 534 | 2 | 14 |
| Month_2/534/Day15.xlsx | xlsx | 0.2351 | 8765 | 534 | 2 | 15 |
| Month_2/534/Day16.xlsx | xlsx | 0.1858 | 6895 | 534 | 2 | 16 |
| Month_2/534/Day17.xlsx | xlsx | 0.1715 | 6332 | 534 | 2 | 17 |
| Month_2/534/Day18.xlsx | xlsx | 0.2029 | 7546 | 534 | 2 | 18 |
| Month_2/534/Day19.xlsx | xlsx | 0.2306 | 8606 | 534 | 2 | 19 |
| Month_2/534/Day20.xlsx | xlsx | 0.2188 | 8166 | 534 | 2 | 20 |
| Month_2/534/Day21.xlsx | xlsx | 0.2333 | 8705 | 534 | 2 | 21 |
| Month_2/534/Day22.xlsx | xlsx | 0.2286 | 8517 | 534 | 2 | 22 |
| Month_2/534/Day23.xlsx | xlsx | 0.2464 | 9221 | 534 | 2 | 23 |
| Month_2/534/Day24.xlsx | xlsx | 0.2345 | 8755 | 534 | 2 | 24 |
| Month_2/534/Day25.xlsx | xlsx | 0.2237 | 8330 | 534 | 2 | 25 |
| Month_2/534/Day26.xlsx | xlsx | 0.2481 | 9298 | 534 | 2 | 26 |
| Month_2/534/Day27.xlsx | xlsx | 0.2223 | 8279 | 534 | 2 | 27 |
| Month_2/534/Day28.xlsx | xlsx | 0.2115 | 7866 | 534 | 2 | 28 |
| Month_2/534/Day29.xlsx | xlsx | 0.2163 | 8066 | 534 | 2 | 29 |
| Month_2/534/Day30.xlsx | xlsx | 0.2416 | 9020 | 534 | 2 | 30 |
| Month_3/425/Day1.xlsx | xlsx | 0.051 | 1750 | 425 | 3 | 1 |
| Month_3/425/Day2.xlsx | xlsx | 0.0615 | 2136 | 425 | 3 | 2 |
| Month_3/425/Day3.xlsx | xlsx | 0.0635 | 2216 | 425 | 3 | 3 |
| Month_3/425/Day4.xlsx | xlsx | 0.0587 | 2035 | 425 | 3 | 4 |
| Month_3/425/Day5.xlsx | xlsx | 0.055 | 1892 | 425 | 3 | 5 |
| Month_3/425/Day6.xlsx | xlsx | 0.0602 | 2084 | 425 | 3 | 6 |
| Month_3/425/Day7.xlsx | xlsx | 0.0608 | 2110 | 425 | 3 | 7 |
| Month_3/425/Day8.xlsx | xlsx | 0.0634 | 2205 | 425 | 3 | 8 |
| Month_3/425/Day9.xlsx | xlsx | 0.0618 | 2142 | 425 | 3 | 9 |
| Month_3/425/Day10.xlsx | xlsx | 0.07 | 2458 | 425 | 3 | 10 |
| Month_3/425/Day11.xlsx | xlsx | 0.0534 | 1827 | 425 | 3 | 11 |
| Month_3/425/Day12.xlsx | xlsx | 0.0478 | 1610 | 425 | 3 | 12 |
| Month_3/425/Day13.xlsx | xlsx | 0.0597 | 2064 | 425 | 3 | 13 |
| Month_3/425/Day14.xlsx | xlsx | 0.0559 | 1925 | 425 | 3 | 14 |
| Month_3/425/Day15.xlsx | xlsx | 0.0467 | 1579 | 425 | 3 | 15 |
| Month_3/425/Day16.xlsx | xlsx | 0.0603 | 2091 | 425 | 3 | 16 |
| Month_3/425/Day17.xlsx | xlsx | 0.0585 | 2021 | 425 | 3 | 17 |
| Month_3/425/Day18.xlsx | xlsx | 0.0587 | 2032 | 425 | 3 | 18 |
| Month_3/425/Day19.xlsx | xlsx | 0.0565 | 1953 | 425 | 3 | 19 |
| Month_3/425/Day20.xlsx | xlsx | 0.0431 | 1446 | 425 | 3 | 20 |
| Month_3/425/Day21.xlsx | xlsx | 0.0497 | 1697 | 425 | 3 | 21 |
| Month_3/425/Day22.xlsx | xlsx | 0.0557 | 1915 | 425 | 3 | 22 |
| Month_3/425/Day23.xlsx | xlsx | 0.0615 | 2135 | 425 | 3 | 23 |
| Month_3/425/Day24.xlsx | xlsx | 0.0605 | 2096 | 425 | 3 | 24 |
| Month_3/425/Day25.xlsx | xlsx | 0.0675 | 2365 | 425 | 3 | 25 |
| Month_3/425/Day26.xlsx | xlsx | 0.0611 | 2125 | 425 | 3 | 26 |
| Month_3/425/Day27.xlsx | xlsx | 0.0566 | 1955 | 425 | 3 | 27 |
| Month_3/425/Day28.xlsx | xlsx | 0.0638 | 2223 | 425 | 3 | 28 |
| Month_3/425/Day29.xlsx | xlsx | 0.0583 | 2013 | 425 | 3 | 29 |
| Month_3/425/Day30.xlsx | xlsx | 0.0629 | 2192 | 425 | 3 | 30 |
| Month_3/425/Day31.xlsx | xlsx | 0.0651 | 2269 | 425 | 3 | 31 |
| Month_3/534/Day1.xlsx | xlsx | 0.2081 | 7738 | 534 | 3 | 1 |
| Month_3/534/Day2.xlsx | xlsx | 0.2243 | 8345 | 534 | 3 | 2 |
| Month_3/534/Day3.xlsx | xlsx | 0.2119 | 7910 | 534 | 3 | 3 |
| Month_3/534/Day4.xlsx | xlsx | 0.1688 | 6233 | 534 | 3 | 4 |
| Month_3/534/Day5.xlsx | xlsx | 0.1975 | 7317 | 534 | 3 | 5 |
| Month_3/534/Day6.xlsx | xlsx | 0.1964 | 7237 | 534 | 3 | 6 |
| Month_3/534/Day7.xlsx | xlsx | 0.2071 | 7700 | 534 | 3 | 7 |
| Month_3/534/Day8.xlsx | xlsx | 0.2008 | 7479 | 534 | 3 | 8 |
| Month_3/534/Day9.xlsx | xlsx | 0.2292 | 8587 | 534 | 3 | 9 |
| Month_3/534/Day10.xlsx | xlsx | 0.2114 | 7871 | 534 | 3 | 10 |
| Month_3/534/Day11.xlsx | xlsx | 0.2135 | 7947 | 534 | 3 | 11 |
| Month_3/534/Day12.xlsx | xlsx | 0.2181 | 8119 | 534 | 3 | 12 |
| Month_3/534/Day13.xlsx | xlsx | 0.1848 | 6864 | 534 | 3 | 13 |
| Month_3/534/Day14.xlsx | xlsx | 0.2224 | 8276 | 534 | 3 | 14 |
| Month_3/534/Day15.xlsx | xlsx | 0.1831 | 6786 | 534 | 3 | 15 |
| Month_3/534/Day16.xlsx | xlsx | 0.1801 | 6687 | 534 | 3 | 16 |
| Month_3/534/Day17.xlsx | xlsx | 0.2277 | 8499 | 534 | 3 | 17 |
| Month_3/534/Day18.xlsx | xlsx | 0.1994 | 7413 | 534 | 3 | 18 |
| Month_3/534/Day19.xlsx | xlsx | 0.2212 | 8246 | 534 | 3 | 19 |
| Month_3/534/Day20.xlsx | xlsx | 0.216 | 8015 | 534 | 3 | 20 |
| Month_3/534/Day21.xlsx | xlsx | 0.223 | 8291 | 534 | 3 | 21 |
| Month_3/534/Day22.xlsx | xlsx | 0.2013 | 7470 | 534 | 3 | 22 |
| Month_3/534/Day23.xlsx | xlsx | 0.1921 | 7157 | 534 | 3 | 23 |
| Month_3/534/Day24.xlsx | xlsx | 0.2076 | 7701 | 534 | 3 | 24 |
| Month_3/534/Day25.xlsx | xlsx | 0.1858 | 6853 | 534 | 3 | 25 |
| Month_3/534/Day26.xlsx | xlsx | 0.2287 | 8529 | 534 | 3 | 26 |
| Month_3/534/Day27.xlsx | xlsx | 0.1981 | 7356 | 534 | 3 | 27 |
| Month_3/534/Day28.xlsx | xlsx | 0.2245 | 8350 | 534 | 3 | 28 |
| Month_3/534/Day29.xlsx | xlsx | 0.2353 | 8770 | 534 | 3 | 29 |
| Month_3/534/Day30.xlsx | xlsx | 0.2192 | 8154 | 534 | 3 | 30 |
| Month_3/534/Day31.xlsx | xlsx | 0.2097 | 7786 | 534 | 3 | 31 |

## Schema (common to every file)

- `Node`: int64
- `Number`: str
- `Route`: str
- `Time`: str

Note: `Route` in the raw files actually encodes **direction** (`up`/`down`), not the bus route number - the real route number comes from the folder name (`425` or `534`). `Node` is an integer stop-sequence index along the route/direction, `Number` is the vehicle registration plate, `Time` is `HH:MM:SS:mmm` time-of-day (no date component).

### First 5 rows (representative file, Month_1/425/Day1.xlsx)

|   Node | Number     | Route   | Time         |
|-------:|:-----------|:--------|:-------------|
|     16 | DL1PC 5377 | up      | 05:24:26:000 |
|     17 | DL1PC 5377 | up      | 05:24:56:000 |
|     18 | DL1PC 5377 | up      | 05:26:36:000 |
|     19 | DL1PC 5377 | up      | 05:27:36:000 |
|     20 | DL1PC 5377 | up      | 05:28:36:000 |

## Latitude / longitude

**No latitude/longitude columns exist anywhere in this dataset** - skipped per request. The only positional information is the integer `Node` (a stop-sequence index), which is already stop-matched (see Node coverage below).

## Where does the date come from?

Checked file names, folder names, cell contents, and Excel workbook metadata (`created`/`modified` properties) for a real calendar date. Findings:

- File/folder names only contain relative labels: `Month_1..3`, `Day1..31`.
- The `Time` column is time-of-day only (`HH:MM:SS:mmm`), no date.
- Workbook metadata (`created`/`modified`) for a sample of files:
- Month_1/425/Day1.xlsx: workbook created=2019-07-09 05:55:27, modified=2019-07-09 05:55:27
- Month_1/425/Day2.xlsx: workbook created=2019-07-09 05:55:28, modified=2019-07-09 05:55:28
- Month_1/425/Day3.xlsx: workbook created=2019-07-09 05:55:29, modified=2019-07-09 05:55:29
- Month_3/534/Day31.xlsx: workbook created=2019-07-09 06:04:04, modified=2019-07-09 06:04:04
  All workbook timestamps fall within the same ~10-minute window on 2019-07-09, regardless of which Month_N/Day_D the file represents - this is the batch export/generation time of the whole archive, not the date the GPS data was collected.

**Conclusion: no real calendar date can be recovered anywhere in this archive.** Only the relative grouping exists: 92 distinct (month, day) combinations, spanning Month_1-Month_3, each with 30 or 31 day-labels. Missing days per (month, route):
- Month_1 / route 425: 31 days present (Day1-Day31), missing: none
- Month_1 / route 534: 31 days present (Day1-Day31), missing: none
- Month_2 / route 425: 30 days present (Day1-Day30), missing: none
- Month_2 / route 534: 30 days present (Day1-Day30), missing: none
- Month_3 / route 425: 31 days present (Day1-Day31), missing: none
- Month_3 / route 534: 31 days present (Day1-Day31), missing: none

## Buses, routes, directions

- Unique vehicle ids (number plates): 202
- Unique routes: 2 (['425', '534'])
- Directions: ['down', 'up']
- Unique buses on route 425: 46
- Unique buses on route 534: 156

## Node coverage per route x direction (= stops)

| route | direction | min node | max node | distinct nodes |
|---|---|---|---|---|
| 425 | down | 1 | 41 | 41 |
| 425 | up | 1 | 37 | 37 |
| 534 | down | 1 | 56 | 56 |
| 534 | up | 1 | 52 | 52 |

## Vehicles and trips per route x direction x day

Total trips across the whole archive: 20839 (a trip = a maximal run of consecutive rows for one vehicle, on one route-day, with the same direction label).

- vehicles per route-direction-day: min 5, median 12.0, max 28
- trips per route-direction-day: min 12, median 43.5, max 102

### Sample of one full trip

|   route | direction   |   month |   day | vehicle_id   |   Node | Time         |
|--------:|:------------|--------:|------:|:-------------|-------:|:-------------|
|     425 | down        |       1 |     1 | DL1PC 5373   |     26 | 07:27:50:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     27 | 07:28:30:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     28 | 07:30:10:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     29 | 07:30:50:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     30 | 07:31:20:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     31 | 07:32:40:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     32 | 07:33:10:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     33 | 07:33:40:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     34 | 07:37:10:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     35 | 07:38:10:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     36 | 07:38:40:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     37 | 07:42:20:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     38 | 07:46:40:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     39 | 07:48:00:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     40 | 07:49:20:000 |
|     425 | down        |       1 |     1 | DL1PC 5373   |     41 | 07:52:50:000 |

## Duplicates and missing values

- Exact duplicate rows (route, month, day, vehicle_id, direction, Node, Time): 0
- Rows with missing values (Node/vehicle_id/direction/Time): 0

## Data quality: node/time ordering within trips

- Trips with `Node` decreasing between consecutive rows (it should be non-decreasing over time for both directions - `up` and `down` each number their own nodes low-to-high in travel order): 71 trips, 72 offending row-to-row steps.
- Trips with time going backwards between consecutive rows: 49 trips, 49 offending steps.
- Trips that skip at least one node (gap > 1 between consecutive nodes): 16625 trips out of 20839 (79.8%), 35117 total skip events.
- Trips crossing midnight (time jumps back by >12h mid-trip): 38

## Time-space diagram

See `reports/figures/00_trips_node_vs_time.png`: x = time of day, y = Node, one line per trip, for route 534, Month_1 Day1, both directions in different colours.

## Stop / schedule / GTFS-like data

**None.** The archive contains only the per-route, per-day `.xlsx` node-crossing files above - no `stops.txt`, timetable, or other GTFS-like file is included. Mapping `Node` indices to real stop names/locations, or getting an actual published schedule, would require a separate Delhi GTFS feed.

## Totals

- Total raw rows: 924,375
- Total trips: 20,839
- Estimated stop-to-stop segments buildable (sum over trips of consecutive node pairs): 903,536


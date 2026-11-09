#!/bin/bash

BFTOOLS=/Users/katherineridley/Downloads/bftools
BFCONVERT="$BFTOOLS/bfconvert"
SHOWINF="$BFTOOLS/showinf"

RAW_DIR="Lifs"
OUT_DIR="images"

mkdir -p "$OUT_DIR"

clean_name () {
    echo "$1" \
    | sed 's/[\/:*?"<>|]/_/g' \
    | sed 's/[[:space:]]\+/_/g' \
    | sed 's/__*/_/g' \
    | sed 's/^_//' \
    | sed 's/_$//'
}

for lif in "$RAW_DIR"/*.lif; do
    [ -e "$lif" ] || continue

    lif_base=$(basename "$lif" .lif)

    echo "Reading OME-XML series names from: $lif"

    tmp_xml="/tmp/${lif_base}_ome.xml"

    "$SHOWINF" -nopix -omexml "$lif" > "$tmp_xml"

    grep '<Image ID=' "$tmp_xml" \
        | sed -E 's/.*Name="([^"]*)".*/\1/' \
        > /tmp/series_names.txt

    n_series=$(wc -l < /tmp/series_names.txt | tr -d ' ')

    echo "Found $n_series series"

    series_index=0

    while IFS= read -r series_name; do
        [ -z "$series_name" ] && series_name="series_${series_index}"

        safe_series_name=$(clean_name "$series_name")

        # Avoid doubling the LIF basename if the internal image name already includes it
        if [[ "$safe_series_name" == "$lif_base"* ]]; then
            out_file="$OUT_DIR/${safe_series_name}.ome.tiff"
        else
            out_file="$OUT_DIR/${lif_base}_${safe_series_name}.ome.tiff"
        fi

        echo "Converting series $series_index / $((n_series-1)):"
        echo "  $series_name"
        echo "  -> $out_file"

        "$BFCONVERT" \
            -overwrite \
            -series "$series_index" \
            "$lif" \
            "$out_file"

        ((series_index++))

    done < /tmp/series_names.txt

done

echo "Done."
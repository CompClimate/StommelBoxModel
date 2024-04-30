import click
import fitz

preset_dict = {
    "data": (170, 125, 0, 0),
    "explain": (10, 10, 245, 0),
}
presets = click.Choice(preset_dict.keys())


@click.command
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option("--preset", type=presets, default="data")
def main(infile, outfile, preset):
    p = preset_dict[preset]
    top_margin, bottom_margin, left_margin, right_margin = p

    doc = fitz.open(infile)
    page = doc[0]

    size = (page.rect.width, page.rect.height)
    rect = fitz.Rect(
        left_margin, top_margin, size[0] - right_margin, size[1] - bottom_margin
    )

    page.set_cropbox(rect)
    doc.save(outfile)


if __name__ == "__main__":
    main()

import pprint
from math import ceil
from typing import List, Optional

import fire
from pylatex import Command, Document, Figure, NoEscape, Package, SubFigure, Tabular


class GridMaker:
    def __init__(
        self,
        infiles: List[str],
        ncols: Optional[int] = 3,
        subcaption_file: Optional[str] = None,
        caption: Optional[str] = None,
        title: Optional[str] = None,
        legend_file: Optional[str] = None,
    ):
        self.infiles = infiles
        self.ncols = ncols
        self.subcaption_file = subcaption_file
        self.caption = caption
        self.title = title
        self.legend_file = legend_file
        self.doc = None

    def __str__(self):
        return pprint.pformat(self.doc)

    def generate_subfigures_tcb(self):
        doc = Document(
            documentclass="standalone", document_options=["varwidth", "border=2pt"]
        )
        doc.packages.append(Package("microtype"))
        doc.packages.append(Package("graphicx"))
        doc.packages.append(
            Package(
                "caption",
                options=[
                    "justification=centering",
                    # "format=plain",
                    "labelformat=parens",
                    # "labelsep=space"
                    "labelsep=period",
                ],
            )
        )
        doc.packages.append(Package("tcolorbox", options=["most"]))
        doc.packages.append(Package("subfig"))

        with doc.create(Figure()):
            if self.title is not None:
                doc.append(NoEscape(r"\begin{center}"))
                doc.append(NoEscape(self.title))
                doc.append(NoEscape(r"\end{center}"))
                doc.append(Command("vspace", arguments=[NoEscape("0.5cm")]))

            doc.append(
                NoEscape(
                    rf"\begin{{tcbitemize}}[raster equal height=rows, raster columns={self.ncols}, raster halign=center, raster every box/.style=blankest]"
                )
            )

            if self.legend_file is not None:
                doc.append(
                    NoEscape(
                        rf"\tcbitem[raster multicolumn={self.ncols},boxed title style={{center}},halign=center]\includegraphics[width=0.33\textwidth]{{{self.legend_file}}}"
                    )
                )

            if self.subcaption_file is not None:
                with open(self.subcaption_file, "r") as f:
                    subcaptions = f.read().splitlines()

                assert (
                    len(self.infiles) == len(subcaptions)
                ), "Each input file has to correspond to one line in the subcaption file."
            else:
                subcaptions = [""] * len(self.infiles)

            for infile, subcaption in zip(self.infiles, subcaptions):
                doc.append(
                    NoEscape(
                        rf"\tcbitem\subfloat[{subcaption}]{{\includegraphics[width=\linewidth,keepaspectratio]{{{infile}}}}}"
                    )
                )

            doc.append(NoEscape(r"\end{tcbitemize}"))
            if self.caption is not None:
                doc.append(Command("caption", arguments=[NoEscape(self.caption)]))

        self.doc = doc
        return self

    def generate_pdf(self, outfile: str, clean_tex: Optional[bool] = False):
        self.doc.generate_pdf(outfile, clean_tex=clean_tex)
        return self


# def generate_tabular_figure(infiles, ncols):
#     doc = Document(document_options=["varwidth"], documentclass="standalone")
#     doc.packages.append(Package("graphicx"))
#     nrows = ceil(len(infiles) / ncols)

#     doc.append(Command("setlength", arguments=[Command("tabcolsep"), "1pt"]))
#     doc.append(Command("renewcommand", arguments=[Command("arraystretch"), "0.9"]))

#     with doc.create(Figure()):
#         doc.append(Command("centering"))
#         with doc.create(Tabular(("c" * ncols))) as tabular:
#             for row in range(nrows):
#                 cells = []

#                 for col in range(ncols):
#                     subfig = SubFigure()
#                     idx = (row * ncols) + col

#                     if idx < len(infiles):
#                         subfig.add_image(infiles[idx], width=NoEscape(r"0.9\textwidth"))

#                         cells.append(subfig)
#                     else:
#                         cells.append(NoEscape(""))

#                 tabular.add_row(cells)

#     return doc


# def generate_subfigures(infiles, ncols):
#     doc = Document(documentclass="standalone", document_options=["varwidth"])
#     doc.packages.append(Package("graphicx"))
#     doc.packages.append(Package("caption", options=["justification=centering"]))

#     with doc.create(Figure()):
#         doc.append(Command("centering"))
#         for i, infile in enumerate(infiles):
#             with doc.create(SubFigure(width=NoEscape(r"0.3\linewidth"))) as subfig:
#                 doc.append(Command("centering"))
#                 subfig.add_image(infile)
#                 subfig.add_caption("hello")
#             if i < len(infiles):
#                 doc.append(Command("hfill"))

#     return doc


# def main(
#     outfile: str,
#     infiles: List[str],
#     ncols: Optional[int] = 3,
#     subcaption_file: Optional[str] = None,
#     caption: Optional[str] = None,
#     title: Optional[str] = None,
#     legend_file: Optional[str] = None,
#     clean_tex: Optional[bool] = False,
# ):
#     doc = generate_subfigures_tcb(
#         infiles=infiles,
#         ncols=ncols,
#         subcaption_file=subcaption_file,
#         caption=caption,
#         title=title,
#         legend_file=legend_file,
#     )
#     doc.generate_pdf(outfile, clean_tex=clean_tex)


if __name__ == "__main__":
    fire.Fire(GridMaker)

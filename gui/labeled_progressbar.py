from tkinter import ttk


class LabeledProgressbar(ttk.Progressbar):
    instance_count = 0

    def __init__(self, parent, **kwargs):
        kwargs.pop("style", None)
        LabeledProgressbar.instance_count += 1

        self.style = ttk.Style()
        self.style_name = f"Text.Horizontal.TProgressbar{LabeledProgressbar.instance_count}"
        self.style.layout(
            self.style_name, [(
                "Horizontal.Progressbar.trough", {
                    "children": [(
                        "Horizontal.Progressbar.pbar", {
                            "side": "left", "sticky": "ns"
                        }
                    )
                    ], "sticky": "nswe"
                }
            ),
                (
                    "Horizontal.Progressbar.label", {
                        "sticky": "nswe"
                    }
                )
            ]
        )

        self.value = kwargs.get("variable", None)
        if self.value is not None:
            self.value.trace("w", self.__update_label)
            self.style.configure(self.style_name, text=f"{self.value.get():.2%}", anchor="center", font=("Consolas", 10))
        else:
            self.style.configure(self.style_name, text=f"0.00%", anchor="center", font=("Consolas", 10))

        super().__init__(parent, style=self.style_name, **kwargs)

    def __update_label(self, a, b, c):
        self.style.configure(self.style_name, text=f"{self.value.get():.2%}")


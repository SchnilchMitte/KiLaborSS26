from IPython.core.display_functions import clear_output
from ipywidgets import IntProgress, Label, HBox, Output
from IPython.display import display


def progressbar(gen, length=None, cleanup=False):
    count = 0
    get_text = lambda: f'{count}/{length} ({count / length * 100:.0f}%)'

    out_context = Output()
    display(out_context)
    pg = IntProgress(min=0, max=length)  # horizontal bar counting up
    label = Label(get_text())  # suffix text
    box = HBox([pg, label])
    box.layout.align_items = 'center'

    with out_context:
        display(box)  # display the bar
        for item in gen:
            count += 1

            yield item
            pg.value = count
            label.value = get_text()
        if cleanup:
            clear_output()  # clean up out_context
        else:
            label.value = get_text() + " Complete."

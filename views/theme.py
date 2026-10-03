SURFACE = "light-dark(#F7F4F1, #1C1C25)"
SECONDARY_TEXT = "light-dark(#606066, #AAAAAA)"

# Red text on carbon is too low contrast, so the selected tab and slider values keep the
# text colour and only the underline and track stay red.
GLOBAL_STYLE = """<style>
[data-testid="stTab"][aria-selected="true"],
[data-testid="stSliderThumbValue"] { color: inherit; }
</style>"""

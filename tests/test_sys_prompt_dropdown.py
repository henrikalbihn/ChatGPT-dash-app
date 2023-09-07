# import pytest
# from dash.testing.application_runners import import_app


# @pytest.mark.parametrize(
#     "dropdown_id, expected_value",
#     [
#         ("sys-prompt", "Academic Essay"),
#         ("output-style", "Outline"),
#     ],
# )
# def test_system_prompt_dropdown(dash_duo, dropdown_id, expected_value):
#     # Import your Dash app
#     app = import_app("app/app.py")  # Replace with the actual module name
#     dash_duo.start_server(app)

#     # Find the dropdown
#     dropdown = dash_duo.find_element(f"#{dropdown_id}")

#     # Check if the default value is correctly set
#     assert dropdown.text == expected_value

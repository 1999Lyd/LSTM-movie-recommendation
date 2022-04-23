

html_string_start = """
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<style>
#left-bar {
  position: fixed;
  display: table-cell;
  top: 100;
  bottom: 10;
  left: 10;
  width: 35%;
  overflow-y: auto;
}
#right-bar {
  position: fixed;
  display: table-cell;
  top: 100;
  bottom: 10;
  right: 10;
  width: 45%;
  overflow-y: auto;
}
</style>
<body>
<center><h1> Top 5 recommendation list </h1></center>
<div id= "left-bar" >
"""


html_string_end = """
</body>
</html>
"""





# define function to add the list element in the html file
def get_count_html(category):
    count_html = """<li> {category_name} </li>"""
    return count_html.format(category_name=category)


# function to calculate the value count



# function to generate the html file from image_class dictionary
# keys will be the path of the images and values will be the class associated to it.
def generate_html(recommendation):

    count_html = ""

    # loop through the keys and add image to the html file

    # loop through the value_counts and add a count of class to the html file
    for ProdId in recommendation:
        count_html += get_count_html(ProdId)

    file_content = html_string_start + """</div> <div id= "right-bar" >""" + count_html + "</div>" + html_string_end
    with open('templates/reclist.html', 'w') as f:
        f.write(file_content)
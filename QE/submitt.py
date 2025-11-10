from aiida.engine import submit
code = load_code('bands@localhost_direct')

builder = code.get_builder()
builder.k
builder.
builder.metadata.options.withmpi = False
builder.metadata.options.resources = {
    'num_machines': 1,

}

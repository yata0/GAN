def write_summary(writer, loss_dict, index):
    for name in loss_dict:
        writer.add_scalar(name,loss_dict[name],index)

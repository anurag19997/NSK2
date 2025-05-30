"""
I/O functions for animal performance analysis.
"""

import os

def save_workbook(wb, root_path, animal_id=None, sum_sheet_count=None):
    """
    Save workbook with animal performance results.
    
    Parameters
    ----------
    wb : Workbook
        Excel workbook object
    root_path : str
        Root directory for saving
    animal_id : str, optional
        Animal ID for filename
    sum_sheet_count : int, optional
        Sheet count for filename
    """
    wb._sheets = sorted(wb._sheets, key=lambda x: x.title)
    
    if animal_id is None:
        if sum_sheet_count is None:
            pth = os.path.join(root_path, 'animal_performance.xlsx')
        else:
            pth = os.path.join(root_path, f'animal_performance_{sum_sheet_count}.xlsx')
    else:
        pth = os.path.join(root_path, f'animal_performance_{animal_id}.xlsx')
    
    print(f'Saving to: {root_path}')
    wb.save(pth)
    wb.close()
    print(f'Saved {pth}')
# Software Engineering Agent Guide

## Project Overview

This is a data analysis dashboard application built with Python and Streamlit. The project consists of multiple files that work together to provide data processing, visualization, and analysis capabilities. This project is designed to test software engineering agent capabilities in code refactoring, debugging, and bug fixing.

## Project Structure

```
├── main.py                 # Main Streamlit application
├── data_processor.py       # Data processing and cleaning logic
├── visualization.py        # Chart generation and visualization
├── utils.py               # Utility functions for file operations
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
└── AGENTS.md              # This guide
```

## Dev Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda for package management

### Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
Create a `requirements.txt` file with:
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
pyyaml>=6.0
requests>=2.31.0
openpyxl>=3.1.0
```

## Running the Application

### Start the Streamlit app
```bash
streamlit run main.py
```

### Development mode
```bash
streamlit run main.py --server.runOnSave true
```

## Testing Instructions

### Manual Testing Checklist
1. **File Upload Testing**
   - Upload various CSV files
   - Test with different file sizes
   - Verify error handling for invalid files

2. **Data Processing Testing**
   - Test missing value cleaning
   - Test outlier removal
   - Test data normalization
   - Verify processing with different data types

3. **Visualization Testing**
   - Generate different chart types
   - Test with various column combinations
   - Verify chart responsiveness
   - Test edge cases (empty data, single column, etc.)

4. **Export Functionality**
   - Test data export in different formats
   - Verify backup creation
   - Test download functionality

### Code Quality Checks
Run these commands to check code quality:

```bash
# Check for syntax errors
python -m py_compile main.py data_processor.py visualization.py utils.py config.py

# Run with debug mode to catch runtime issues
python -c "import main; print('Import successful')"
```

### Performance Testing
- Test with large datasets (>10MB)
- Monitor memory usage during processing
- Test concurrent user access
- Verify chart rendering performance

## Code Architecture

### Main Components

1. **main.py**: Entry point and UI logic
   - Handles file uploads
   - Manages session state
   - Coordinates between components

2. **data_processor.py**: Core data processing
   - Data cleaning and transformation
   - Statistical analysis
   - Outlier detection

3. **visualization.py**: Chart generation
   - Multiple chart types support
   - Interactive plotly charts
   - Dashboard layouts

4. **utils.py**: Helper functions
   - File I/O operations
   - Data validation
   - Memory optimization

5. **config.py**: Configuration management
   - Application settings
   - Environment-specific configurations
   - Default values

## Common Issues and Solutions

### Performance Issues
- Large datasets may cause memory issues
- Chart rendering can be slow with many data points
- File processing may timeout with very large files

### Data Quality Issues
- Missing values may not be handled properly in all cases
- Outlier detection may be too aggressive
- Data type inference may fail for mixed columns

### Visualization Issues
- Charts may not display correctly with certain data combinations
- Color palettes may not be sufficient for many categories
- Responsive design may not work on all screen sizes

## Development Guidelines

### Code Style
- Follow PEP 8 standards
- Use type hints where appropriate
- Add docstrings to all functions
- Keep functions focused and small

### Error Handling
- Always handle file I/O errors
- Validate user inputs
- Provide meaningful error messages
- Log errors appropriately

### Testing
- Test edge cases thoroughly
- Verify data integrity after processing
- Test with various data sizes and types
- Validate chart outputs

## Debugging Tips

### Common Debugging Scenarios
1. **Data Loading Issues**
   - Check file encoding
   - Verify file format
   - Examine column names and data types

2. **Processing Errors**
   - Validate input data
   - Check for missing dependencies
   - Monitor memory usage

3. **Visualization Problems**
   - Verify data format for charts
   - Check column data types
   - Ensure data is not empty

### Logging
- Enable debug logging in config
- Check Streamlit logs for errors
- Monitor console output

## Contribution Guidelines

### Code Quality Standards
- Write clean, readable code
- Add appropriate comments
- Follow existing code patterns
- Test thoroughly before submitting

### Performance Considerations
- Optimize for large datasets
- Minimize memory usage
- Cache expensive operations
- Use efficient algorithms

### Documentation
- Update this guide when making significant changes
- Document new features and functions
- Provide examples for complex operations

## Known Limitations

1. **Scalability**: Not optimized for very large datasets (>1GB)
2. **File Formats**: Limited support for complex Excel features
3. **Visualization**: Some chart types may not work with all data types
4. **Memory**: No streaming support for large file processing
5. **Security**: Limited input validation and sanitization

## Future Improvements

1. Add support for more file formats
2. Implement data streaming for large files
3. Add more advanced visualization options
4. Improve error handling and user feedback
5. Add automated testing framework
6. Implement user authentication
7. Add data export scheduling
8. Improve mobile responsiveness
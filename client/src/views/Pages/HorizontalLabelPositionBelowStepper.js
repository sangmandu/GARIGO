import React from 'react';
import {DropzoneArea} from 'material-ui-dropzone';
import ImageUploader from "react-images-upload";
import {makeStyles} from '@material-ui/core/styles';
import Stepper from '@material-ui/core/Stepper';
import Step from '@material-ui/core/Step';
import StepLabel from '@material-ui/core/StepLabel';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import axios from "axios";

class MyUploader extends React.Component {
    constructor(props) {
        super(props);
        this.state = {pictures: []};
        this.onDrop = this.onDrop.bind(this);
    }

    onDrop(pictureFiles, pictureDataURLs) {
        this.setState({
            pictures: this.state.pictures.concat(pictureFiles)
        });
    }

    render() {
        const onClickSendImage = (e) => {
            e.preventDefault();
            const formData = new FormData();

            this.state.pictures.map(file => {
                formData.append("photos", file);
            });

            axios({
                method: "post",
                url: 'http://localhost:8000/media/',
                data: formData,
                headers: {"Content-Type": "multipart/form-data", Authorization: localStorage.getItem("access_token")}
            });
        };
        return (
            <div>
                <DropzoneArea
                    acceptedFiles={['image/*']}
                    dropzoneText={"Drag and drop an image here or click"}
                    onChange={(files) => {
                        console.log('Files:', files);
                        this.setState({
                            pictures: files
                        });
                    }}
                />
                <Button onClick={onClickSendImage}>사진 전송</Button>
            </div>
        );
    }
}

const useStyles = makeStyles((theme) => ({
    root: {
        width: '100%',
    },
    backButton: {
        marginRight: theme.spacing(1),
    },
    instructions: {
        marginTop: theme.spacing(1),
        marginBottom: theme.spacing(1),
    },
}));

function getSteps() {
    return ['모자이크 처리하지않을 얼굴 사진 올려주세요', '모자이크 처리할 동영상을 업로드해주세요.', '완성되었습니다'];
}

function getStepContent(stepIndex) {
    switch (stepIndex) {
        case 0:
            return '사진 업로드 대기 중...';
        case 1:
            return 'What is an ad group anyways?';
        case 2:
            return 'This is the bit I really care about!';
        default:
            return 'Unknown stepIndex';
    }
}

export default function HorizontalLabelPositionBelowStepper() {
    const classes = useStyles();
    const [activeStep, setActiveStep] = React.useState(0);
    const steps = getSteps();

    const handleNext = () => {
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
    };

    const handleBack = () => {
        setActiveStep((prevActiveStep) => prevActiveStep - 1);
    };

    const handleReset = () => {
        setActiveStep(0);
    };


    return (
        <div className={classes.root}>
            <Stepper activeStep={activeStep} alternativeLabel>
                {steps.map((label) => (
                    <Step key={label}>
                        <StepLabel>{label}</StepLabel>
                    </Step>
                ))}
            </Stepper>
            <div>
                {activeStep === steps.length ? (
                    <div>
                        <Typography className={classes.instructions}>All steps completed</Typography>
                        <Button onClick={handleReset}>Reset</Button>
                    </div>
                ) : (
                    <div>
                        <Typography className={classes.instructions}>{getStepContent(activeStep)}</Typography>
                        <div>
                            <Button
                                disabled={activeStep === 0}
                                onClick={handleBack}
                                className={classes.backButton}
                            >
                                Back
                            </Button>
                            <Button variant="contained" color="primary" onClick={handleNext}>
                                {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
                            </Button>
                        </div>
                    </div>
                )}
            </div>
            <MyUploader/>
        </div>
    );
}
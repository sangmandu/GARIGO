import React from 'react';
import {DropzoneArea} from 'material-ui-dropzone';
import {makeStyles} from '@material-ui/core/styles';
import Stepper from '@material-ui/core/Stepper';
import Step from '@material-ui/core/Step';
import StepLabel from '@material-ui/core/StepLabel';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import axios from "axios";

import LinearProgress from '@material-ui/core/LinearProgress';
import {Link} from "@material-ui/core";

export function LinearDeterminate(props) {
    const classes = useStyles();
    const [progress, setProgress] = React.useState(0);

    React.useEffect(() => {
        const timer = setInterval(() => {
            setProgress((oldProgress) => {
                if (oldProgress === 100) {
                    props.changeIsCompleted();
                    clearInterval(timer);
                }
                const diff = Math.random() * 10;
                return Math.min(oldProgress + diff, 100);
            });
        }, 500);

        return () => {
            clearInterval(timer);
        };
    }, []);

    return (
        <div className={classes.root}>
            <LinearProgress variant="determinate" value={progress}/>
        </div>
    );
}

class MyUploader extends React.Component {

    constructor(props) {
        super(props);
        this.state = {pictures: []};
        this.onDrop = this.onDrop.bind(this);
    }

    reset = () => {
        this.setState({key: 0, files: []});
    };


    onDrop(pictureFiles, pictureDataURLs) {
        this.setState({
            pictures: this.state.pictures.concat(pictureFiles)
        });
    }

    render() {
        const onClickSendImage = async (e) => {
            e.preventDefault();
            const formData = new FormData();

            this.state.pictures.map(file => {
                formData.append("photos", file);
            });

            console.log(this.props.pid)

            if (!!this.props.pid) {
                formData.append('pid', this.props.pid);
            }

            let response = await axios({
                method: "post",
                url: 'http://localhost:8000/media/',
                data: formData,
                headers: {
                    "Content-Type": "multipart/form-data",
                    Authorization: localStorage.getItem("access_token")
                }
            });

            if (!(!!this.props.pid)) {
                this.props.changePid(response.data['pid']);
            } else {
            }

            this.reset();

            if (!!this.props.completeImageUpload) {
                this.props.completeImageUpload();
            }

            console.log(formData);

            console.log(this.state.pictures)
        }

        return (
            <div key={this.state.key} style={{display: 'flex', flexDirection: 'column'}}>
                { 
                    <DropzoneArea
                        dropzoneText={"Drag and drop an image here or click"}
                        filesLimit={100}
                        maxFileSize={300000000}
                        fileObjects={this.state.pictures}
                        onChange={(files) => {
                            console.log('Files:', files);
                            this.setState({
                                pictures: files
                            });
                        }}
                    />
                }
                <Button
                    style={{display: 'flex', alignSelf: 'center', justifySelf: 'center', flex: 1, marginTop: '1rem'}}
                    variant="contained" color="primary" onClick={onClickSendImage}>사진 전송</Button>
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

function getStepContent(stepIndex) {
    switch (stepIndex) {
        case 0:
            return '사진 업로드 대기 중...';
        case 1:
            return '영상 업로드 대기 중...';
        case 2:
            return '';
        default:
            return 'Unknown stepIndex';
    }
}

export default function HorizontalLabelPositionBelowStepper() {
    const classes = useStyles();
    const [activeStep, setActiveStep] = React.useState(0);
    const [pid, setPid] = React.useState(null);
    const [disable0, setDisable0] = React.useState(true);
    const [isCompleted, setIsCompleted] = React.useState(false);
    const steps = getSteps();

    function getSteps() {
        return ['모자이크 처리하지않을 얼굴 사진 올려주세요', '모자이크 처리할 동영상을 업로드해주세요.', isCompleted ? '완성되었습니다' : '모자이크 처리 작업중'];
    }

    const handleNext = () => {
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
    };

    const handleBack = () => {
        setActiveStep((prevActiveStep) => prevActiveStep - 1);
    };

    const handleReset = () => {
        setActiveStep(0);
    };

    const changePid = (newPid) => {
        setPid(newPid);
    };

    const changeIsCompleted = () => {
        setIsCompleted(true);
    }

    const completeImageUpload = () => {
        setDisable0(false);
    }

    const renderSwitch = (activeStep) => {
        switch (activeStep) {
            case 0:
                return (<MyUploader pid={pid} completeImageUpload={completeImageUpload} changePid={changePid}/>);
            case 1:
                return (<MyUploader pid={pid} changePid={changePid}/>);
            case 2:
                return (<LinearDeterminate changeIsCompleted={changeIsCompleted}/>);
        }
    }


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
                            <Button variant="contained" color="primary" onClick={handleNext} disabled={disable0}>
                                {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
                            </Button>
                        </div>
                    </div>
                )}
            </div>
            {
                renderSwitch(activeStep)
            }

            {
                isCompleted ? (<div style={{ marginTop:'2rem'}}><Link
                    style={{fontSize: '2rem', marginTop: '2rem', padding: '2rem'}}
                    href="https://garigo.s3.ap-northeast-2.amazonaws.com/fe895e07-ae01-4c09-846c-f58dcab29b41.mp4">
                    Download
                </Link></div>) : ''
            }
        </div>
    );
}